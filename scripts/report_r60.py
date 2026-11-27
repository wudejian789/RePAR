"""Audit complete dense-index conditions and report seed-matched scores and cost."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
import pickle
import re

import numpy as np

from repar.benchmarks.math_exam import aggregate_grades, public_question, render_problem, score_choice
from repar.benchmarks.v3_suite import sha, unique
from repar.evaluation import extract_answer
from repar.io import digest, load_public_samples, read_json, read_jsonl, write_json
from repar.v4.workflow import FOCUSED_PLAN_PROMPT


ROOT = Path("artifacts/repar_v4_r60_20261126")
BASE = Path("artifacts/repar_v4_r27_historical_seeds_20261126/summary.json")
HISTORICAL = Path("artifacts/repar_v4_r27_20261120")


def check_indices(folder):
    indexes = read_json(folder/"indexes.json")
    if len(indexes) != len({i["context_sha256"] for i in indexes}) or len(indexes) != 126:
        raise ValueError("Expected 126 unique sources")
    for item in indexes:
        with Path(item["path"]).open("rb") as stream:
            index = pickle.load(stream)
        limit = index.stats["chunk_max_tokens"]
        assert limit == item["chunk_max_tokens"]
        assert digest(index.context) == item["context_sha256"]
        assert index.spans[0][0] == 0 and index.spans[-1][1] == len(index.context)
        assert all(a[1] == b[0] for a, b in zip(index.spans, index.spans[1:]))
        assert all(index.context[a:b] == text for (a, b), text in zip(index.spans, index.texts, strict=True))
        assert all(0 <= count <= limit for count in index.chunk_tokens)
        assert len(index.chunk_tokens) == len(index.spans)
        assert not {"postings", "vectorizer"} & vars(index).keys()
        assert index.matrix.shape == (len(index.spans), 768) and np.isfinite(index.matrix).all()
        assert np.allclose(np.linalg.norm(index.matrix, axis=1), 1, atol=1e-5)
        assert index.tree_matrix.shape == (len(index.nodes), 768)
        assert set(index.entity_graph.extractor["models"]) <= {"zh_core_web_sm", "en_core_web_sm"}
        for proposition in index.entity_graph.propositions:
            assert index.context[proposition["start"]:proposition["end"]] == proposition["text"]
            lo, hi = index.spans[int(proposition["passage_id"][1:])]
            assert lo <= proposition["start"] < proposition["end"] <= hi
    return {item["context_sha256"]: item for item in indexes}


def audit_seed(run, indexes, reference, version="r60"):
    protocol = read_json(run/"protocol.json")
    config, solver = (protocol["identity"][name] for name in ("config", "solver"))
    seed = solver["seed"]
    assert solver["agent_version"] == f"react-v4-{version}"
    for name, checksum in protocol["identity"]["files"].items():
        assert sha(run/"source"/name) == checksum
    for field in ("data", "references"):
        for name, checksum in protocol["identity"][field].items():
            assert sha(name) == checksum
    assert protocol["identity"]["data"] == reference["identity"]["data"]
    assert config["judge"] == reference["identity"]["config"]["judge"]
    assert all(solver[key] == value for key, value in reference["identity"]["solver"].items()
               if key not in {"index_version", "chunk_words", "graph_model", "seed", "deployment_id",
                              "agent_version", "source_policy", "notes"})
    for role in ("solver", "judge"):
        for key in ("model_path", "model_alias", "build_info", "total_slots", "chat_template",
                    "default_generation_settings"):
            assert protocol["deployments"][role]["props"][key] == reference["deployments"][role]["props"][key]
    math_questions = {row["id"]: row for row in read_jsonl(config["math_questions"])}
    long_questions = {row["_id"]: row for row in load_public_samples(config["longbench_data"])}
    stats = {}
    for dataset, questions in (("math", math_questions), ("longbench", long_questions)):
        path = run/dataset/"predictions.jsonl"
        rows = unique(path, expected=questions)
        assert len(rows) == (19 if dataset == "math" else 125)
        assert read_json(run/dataset/"generation_manifest.json")["predictions_sha256"] == sha(path)
        old_run = HISTORICAL if seed == 42 else BASE.parent/"runs"/f"seed_{seed}"
        old_rows = unique(old_run/dataset/"predictions.jsonl", expected=questions)
        grades = unique(run/"math/grades.jsonl", expected=questions) if dataset == "math" else {}
        scores, totals, stages = {}, Counter(), {"control": Counter(), "final": Counter()}
        math_index = read_json(run/"math/index_reuse.json") if dataset == "math" else None
        for qid, row in rows.items():
            assert row["status"] == "ok" and row["agent_version"] == solver["agent_version"]
            assert row["task"] == old_rows[qid]["task"]
            assert row["context_sha256"] == old_rows[qid]["context_sha256"]
            metadata = math_index or row["index_reuse"]
            assert metadata["path"] == indexes[row["context_sha256"]]["path"]
            if version == "r61":
                if dataset == "math":
                    question = public_question(questions[qid])
                    task_text = render_problem(question, readable_figures=True)
                    anchor = "\n".join([question["question_stem"], *(x["question"] for x in question["subquestions"])])
                else:
                    question = questions[qid]
                    anchor = question["question"]
                    task_text = "\n".join([anchor, *(question["choice_" + key] for key in "ABCD")])
            if dataset == "math":
                grade = grades[qid]
                assert grade["response_sha256"] == digest(row["response"])
                q = questions[qid]
                if q["question_type"] in ("single_choice", "multiple_choice"):
                    checked = score_choice(q, row["response"])
                else:
                    checked = aggregate_grades(q, grade["grades"])
                assert checked["score"] == grade["score"]
                scores[qid] = grade["score"]
            else:
                pred = extract_answer(row["response"])
                assert pred == row["pred"] and row["correct"] == (pred == questions[qid]["answer"])
                scores[qid] = int(row["correct"])
            for step in row["trace"]:
                query = step.get("plan", {}).get("query")
                if query:
                    totals["plan_queries"] += 1
                    totals["plan_queries_with_han"] += bool(re.search(r"[\u3400-\u9fff]", query))
                totals["tool_calls"] += len(step["executed"])
                for call in step["executed"]:
                    if call["name"] == "tree_search" and call["observation"]["status"] in ("ok", "no_match"):
                        events = call["observation"]["query_execution"]
                        assert all(e["candidate_generation"]["method"] == "embeddinggemma_cosine" for e in events)
                        totals["dense_channels"] += len(events)
                        if version == "r61":
                            if call["parameters"].get("anchor_query"):
                                assert call["parameters"]["anchor_query"] == anchor
                            for event in events:
                                if event["parameters"]["mode"] == "hybrid":
                                    exact = event["candidate_generation"]["exact_dense_ids"]
                                    assert event["source_ids"] == exact[:len(event["source_ids"])]
                                    totals["exact_scoped_channels"] += 1
            for call in row["calls"]:
                request = read_json(run/"cache/solver"/(call["key"]+".json"))["request"]
                assert request["seed"] == seed and request["temperature"] == solver["temperature"]
                usage = call["response"]["usage"]
                totals["model_calls"] += 1
                totals["cached_calls"] += call["cached"]
                totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
                totals["completion_tokens"] += usage.get("completion_tokens", 0)
                final = call["tag"] in ("math_repar_answer", "v2_repar_answer")
                stage = stages["final" if final else "control"]
                stage["input"] += usage.get("prompt_tokens", 0)
                stage["output"] += usage.get("completion_tokens", 0)
                totals["control_repairs"] += call["tag"].endswith("_repair")
                if version == "r61" and call["tag"].startswith("v4_plan_") and not call["tag"].endswith("_repair"):
                    assert request["messages"][0]["content"] == FOCUSED_PLAN_PROMPT
                    assert json.loads(request["messages"][1]["content"])["task"] == task_text
                if final:
                    old_call = old_rows[qid]["calls"][-1]
                    prior = read_json(old_run/"cache/solver"/(old_call["key"]+".json"))["request"]
                    assert request == {**prior, "messages": request["messages"]}
                    totals["final_length"] += call["response"]["choices"][0]["finish_reason"] == "length"
        stats[dataset] = {"score": sum(scores.values()), "questions": len(rows), "per_question": scores, "stages": stages,
                          "terminations": dict(Counter(row["termination"] for row in rows.values())),
                          "mean_online_seconds": mean(row["seconds"] for row in rows.values()), **totals}
    write_json(run/f"audit_{version}.json", {"passed": True, "stats": stats})
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[192, 384, 768, 1024])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(42, 43, 44), default=[42])
    parser.add_argument("--version", choices=("r60", "r61"), default="r60")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide distinct seeds")
    root = ROOT if args.version == "r60" else Path("artifacts/repar_v4_r61_20261126")
    baseline = read_json(BASE)
    results = {"baseline": {}, "conditions": {}}
    for dataset in ("math", "longbench"):
        scores = {seed: baseline[dataset]["versions"][f"r27_seed_{seed}"]["score"] for seed in args.seeds}
        results["baseline"][dataset] = {"mean": mean(scores.values()), "seeds": scores}
    reference = read_json(HISTORICAL/"protocol.json")
    previous = read_json(ROOT/"comparison.json")["conditions"] if args.version == "r61" else {}
    for size in args.sizes:
        folder = root/f"tokens_{size}"
        indexes = check_indices(ROOT/f"tokens_{size}")
        parent = read_json(ROOT/f"tokens_{size}"/"seed_42"/"protocol.json") if args.version == "r61" else None
        if parent:
            for seed in args.seeds:
                references = read_json(folder/f"seed_{seed}"/"protocol.json")["identity"]["references"]
                assert all(references.get(path) == checksum
                           for path, checksum in parent["identity"]["references"].items() if path.endswith(".pkl"))
        seeds = {seed: audit_seed(folder/f"seed_{seed}", indexes, reference, args.version) for seed in args.seeds}
        averages = {dataset: mean(stats[dataset]["score"] for stats in seeds.values())
                    for dataset in ("math", "longbench")}
        variability = {dataset: (stdev(stats[dataset]["score"] for stats in seeds.values()) if len(seeds) > 1 else None)
                       for dataset in averages}
        deltas = {dataset: {seed: stats[dataset]["score"]-results["baseline"][dataset]["seeds"][seed]
                            for seed, stats in seeds.items()}
                  for dataset in averages}
        chunk_stats = {"math": next(iter(indexes.values()))["leaves"],
                       "longbench_average": mean(i["leaves"] for i in list(indexes.values())[1:])}
        results["conditions"][size] = {
            "seeds": seeds, "average": averages, "sample_stdev": variability, "seed_matched_deltas": deltas,
            "mean_delta": {dataset: mean(values.values()) for dataset, values in deltas.items()},
            "chunks": chunk_stats,
            "index_seconds_summed": {key: sum(item[key] for item in indexes.values())
                                     for key in ("build_seconds", "embedding_seconds", "graph_build_seconds")},
        }
        if parent:
            prior = previous.get(str(size), {}).get("seeds", {})
            old = {}
            for seed in seeds:
                if str(seed) in prior:
                    old[str(seed)] = prior[str(seed)]
                else:
                    audit = read_json(ROOT/f"tokens_{size}"/f"seed_{seed}"/"audit_r60.json")
                    assert audit["passed"]
                    old[str(seed)] = audit["stats"]
            results["conditions"][size]["mean_delta_vs_r60"] = {
                dataset: mean(seeds[seed][dataset]["score"]-old[str(seed)][dataset]["score"] for seed in seeds)
                for dataset in averages}
        print(size, averages, chunk_stats, flush=True)
    default_name = "comparison.json" if args.seeds == [42, 43, 44] else (
        "comparison_seed_" + "_".join(map(str, args.seeds)) + ".json")
    write_json(args.output or root/default_name, results)


if __name__ == "__main__":
    main()
