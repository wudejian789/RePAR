"""Audit v5 coverage, request settings, source provenance and paired results."""
import argparse
from collections import Counter
from hashlib import file_digest
import json
from math import comb
from pathlib import Path

from repar.benchmarks.math_exam import aggregate_grades, public_question, render_problem, score_choice, stitch_context
from repar.benchmarks.v3_suite import sha, unique
from repar.evaluation import extract_answer, qa_text
from repar.client import sampling_parameters
from repar.io import digest, load_public_samples, read_json, read_jsonl, write_json


def verify_hash(path, expected):
    with Path(path).open("rb") as stream:
        assert file_digest(stream, "sha256").hexdigest() == expected, path


def audit(run, reference, backbone_reference=None):
    protocol = read_json(run/"protocol.json")
    identity = protocol["identity"]
    config, solver = identity["config"], identity["solver"]
    prior_protocol = read_json(reference/"protocol.json")
    backbone_protocol = read_json(backbone_reference/"protocol.json") if backbone_reference else prior_protocol
    if backbone_reference:
        assert config["backbone_reference"] == backbone_reference.as_posix()
        assert config["workflow_reference"] == reference.as_posix()
        assert (backbone_reference/"protocol.json").as_posix() in identity["references"]
    for name, expected in identity["files"].items():
        verify_hash(run/"source"/name, expected)
    for name, expected in {**identity["data"], **identity["references"]}.items():
        verify_hash(name, expected)
    assert identity["data"] == prior_protocol["identity"]["data"]
    assert config["judge"] == prior_protocol["identity"]["config"]["judge"]
    assert config["dense_index_directory"] == prior_protocol["identity"]["config"]["dense_index_directory"]
    connection_keys = {"base_url", "model", "context_window"} if backbone_reference else set()
    for key in connection_keys:
        assert solver[key] == backbone_protocol["identity"]["solver"][key], key
    for key, value in prior_protocol["identity"]["solver"].items():
        if key not in {"agent_version", "deployment_id", "notes"} | connection_keys:
            assert solver[key] == value, key
    for role in ("solver", "judge"):
        props = protocol["deployments"][role]["props"]
        old = (backbone_protocol if role == "solver" else prior_protocol)["deployments"][role]["props"]
        for key in ("model_path", "model_alias", "build_info", "total_slots", "default_generation_settings", "chat_template"):
            assert props[key] == old[key], (role, key)

    questions = {"math": {q["id"]: q for q in read_jsonl(config["math_questions"])},
                 "longbench": {q["_id"]: q for q in load_public_samples(config["longbench_data"])}}
    math_text, _ = stitch_context(config["math_textbook"])
    math_index = read_json(run/"math/index_reuse.json")
    scores = {}
    for name, public in questions.items():
        sampling = config.get("sampling_by_dataset", {}).get(name, {})
        expected_sampling = {"temperature": solver["temperature"],
                             **sampling_parameters({"sampling": sampling})}
        path = run/name/"predictions.jsonl"
        predictions = unique(path, expected=public)
        old_rows = unique(reference/name/"predictions.jsonl", expected=public)
        grades = unique(run/"math/grades.jsonl", expected=public) if name == "math" else {}
        assert read_json(path.parent/"generation_manifest.json")["predictions_sha256"] == sha(path)
        assert len(predictions) == (19 if name == "math" else 125)
        totals, terminations, tools = Counter(), Counter(), Counter()
        draft_tokens, final_tokens = Counter(), Counter()
        paired = []
        for qid, row in predictions.items():
            old_row = old_rows[qid]
            assert row["status"] == "ok" and row["agent_version"] == "react-v5"
            assert (row["task"], row["context_sha256"]) == (old_row["task"], old_row["context_sha256"])
            source = math_text if name == "math" else public[qid]["context"]
            index = math_index if name == "math" else row["index_reuse"]
            assert index["path"] in identity["references"] and index["reused"]
            assert digest(source) == index["context_sha256"] == row["context_sha256"]
            for record in row["evidence_records"]:
                assert source[record["start"]:record["end"]] == record["text"]
                totals["source_slices_checked"] += 1
            prior, *control, final = row["calls"]
            assert prior["tag"] == "v5_direct_answer"
            assert final["tag"] == old_row["calls"][-1]["tag"]
            draft = row["state"]["direct_answer"]
            draft_message = prior["response"]["choices"][0]["message"].get("content") or ""
            assert draft == draft_message.strip()
            source_task = (render_problem(public_question(public[qid]), readable_figures=True)
                           if name == "math" else qa_text(public[qid]))
            requests = [read_json(run/"cache/solver"/(call["key"]+".json"))["request"] for call in row["calls"]]
            assert requests[0]["messages"][-1] == {"role": "user", "content": source_task}
            assert requests[0]["max_tokens"] == solver["direct_answer_max_tokens"]
            assert "grammar" not in requests[0] and "response_format" not in requests[0]
            old_final = read_json(reference/"cache/solver"/(old_row["calls"][-1]["key"]+".json"))["request"]
            expected_final = {**old_final, "messages": requests[-1]["messages"]}
            if backbone_reference:
                expected_final.update(model=solver["model"], **expected_sampling)
            assert requests[-1] == expected_final
            assert json.loads(requests[-1]["messages"][-1]["content"].split("\n\n", 1)[0])["provisional_answer"] == draft
            for call, request in zip(control, requests[1:-1], strict=True):
                if not call["tag"].endswith("_repair"):
                    assert json.loads(request["messages"][1]["content"])["provisional_answer"] == draft
            for call, request in zip(row["calls"], requests, strict=True):
                usage = call["response"]["usage"]
                assert request["model"] == solver["model"] and request["seed"] == solver["seed"]
                assert all(request[key] == value for key, value in expected_sampling.items())
                totals["model_calls"] += 1
                totals["cached_calls"] += call["cached"]
                totals["control_repairs"] += call["tag"].endswith("_repair")
                for token in ("prompt_tokens", "completion_tokens"):
                    totals[token] += usage[token]
                    if call is prior:
                        draft_tokens[token] += usage[token]
                    if call is final:
                        final_tokens[token] += usage[token]
            event_rows = read_jsonl(run/name/"events"/(qid+".jsonl"))
            assert event_rows and event_rows[0] == {"stage": "direct_answer", "answer": draft}
            seen = set()
            for event in event_rows[1:]:
                if event["stage"] == "observation":
                    assert event["executed"]
                    seen.add(event["round"])
                if event["stage"] == "reflection":
                    assert event["round"] in seen
            for step in row["trace"]:
                for call in step["executed"]:
                    tools[call["name"]] += 1
                    for record in call["visible_observation"].get("source_views", []):
                        assert source[record["start"]:record["end"]] == record["text"]
                        totals["source_slices_checked"] += 1
                    if call["name"] == "tree_search" and call["observation"]["status"] in {"ok", "no_match"}:
                        for query in call["observation"]["query_execution"]:
                            candidates = query["candidate_generation"]
                            assert candidates["method"] == "embeddinggemma_cosine"
                            if query["parameters"]["mode"] == "hybrid":
                                assert query["source_ids"] == candidates["exact_dense_ids"][:len(query["source_ids"])]
            terminations[row["termination"]] += 1
            totals["no_tool_runs"] += not any(step["executed"] for step in row["trace"])
            totals["empty_drafts"] += not draft
            totals["draft_length"] += prior["response"]["choices"][0]["finish_reason"] == "length"
            totals["final_length"] += final["response"]["choices"][0]["finish_reason"] == "length"
            if name == "math":
                grade, question = grades[qid], public[qid]
                assert grade["response_sha256"] == digest(row["response"])
                checked = (score_choice(question, row["response"]) if question["question_type"] in {"single_choice", "multiple_choice"}
                           else aggregate_grades(question, grade["grades"]))
                assert checked["score"] == grade["score"]
                totals["score"] += grade["score"]
                totals["fully_correct"] += grade["fully_correct"]
            else:
                assert extract_answer(row["response"]) == row["pred"]
                assert row["correct"] == (row["pred"] == public[qid]["answer"])
                totals["score"] += row["correct"]
                paired.append((old_row["correct"], row["correct"]))
        scores[name] = {**totals, "questions": len(predictions), "draft_usage": draft_tokens, "final_usage": final_tokens,
                        "terminations": terminations, "tool_calls_by_name": tools}
        if paired:
            gained = paired.count((False, True))
            lost = paired.count((True, False))
            discordant = gained + lost
            p_value = min(1, 2 * sum(comb(discordant, k) for k in range(min(gained, lost)+1)) / 2**discordant)
            scores[name].update(gained=gained, lost=lost, mcnemar_exact_p=p_value)
    write_json(run/"audit_v5.json", {"passed": True, "reference": reference.as_posix(),
              **({"backbone_reference": backbone_reference.as_posix()} if backbone_reference else {}), "stats": scores})
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/repar_v5_20261126/seed_42"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/repar_v4_r61_20261126/tokens_768/seed_42"))
    parser.add_argument("--backbone-reference", type=Path, help="Frozen deployment used for a cross-backbone run")
    args = parser.parse_args()
    audit(args.run, args.reference, args.backbone_reference)
