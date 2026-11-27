"""Export audited cases, exact evidence and archived knowledge indexes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from repar.index import KnowledgeIndex
from repar.io import digest, read_json, read_jsonl
from repar.benchmarks.math_exam import public_question, stitch_context, task_text
from webserver.frozen_data import (
    HISTORICAL_MATH_RUN, INDEX_METHODS, MATH_BASELINES, RUNS, V5_METHOD, file_digest, modern_result, read_condition,
    read_historical_math,
)
from webserver.knowledge_data import bm25_data, load_frozen_index, tree_graph_data

METHODS = {"direct": "直接回答", "bm25": "BM25 RAG", "repar": "RePAR", "repar_tree": "RePAR · 禁用图"}

LONG_BENCH_FINAL_TAGS = {
    "direct": "direct", "bm25": "bm25_answer", "repar": "repar_answer",
    "repar_tree": "repar_tree_answer", "repar_v2": "v2_repar_answer",
    "repar_tree_v2": "v2_repar_tree_answer", "direct_27b": "direct", "bm25_27b": "bm25_answer",
    "repar_r27_08b": "v2_repar_answer", "repar_r27_27b": "v2_repar_answer",
    V5_METHOD: "v2_repar_answer",
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)


def export_final_inputs(root=ROOT, output=None):
    root = Path(root)
    output = Path(output or root / "webserver/data")
    artifacts = root / "artifacts"
    modern_runs = {method: root / path for method, (_, path) in RUNS.items()}
    counts = {}
    for dataset, directory in (("longbench", output), ("math", output / "math")):
        overview = read_json(directory / "overview.json")
        solver_by_method = {}
        for method in overview["methods"]:
            if method in modern_runs:
                run = modern_runs[method]
                solver = read_json(run / "protocol.json")["identity"]["solver"]
                cache = run / "cache/solver"
            elif method in MATH_BASELINES:
                run = root / HISTORICAL_MATH_RUN
                solver = read_json(run / "protocol.json")["identity"]["resolved_solver"]
                cache = run / "cache/solver"
            else:
                run = artifacts / ("experiment_v2" if method.endswith("_v2") else "experiment")
                solver = read_json(run / "run_metadata.json")["config"]
                cache = artifacts / "cache/llm"
            if dataset == "math":
                family = method.split("_", 1)[0]
                tag = f"math_{family}_answer"
            else:
                tag = LONG_BENCH_FINAL_TAGS[method]
            solver_by_method[method] = (solver, cache, tag)
        counts[dataset] = 0
        for entry in overview["cases"]:
            cid = entry["id"]
            case = read_json(directory / "cases" / f"{cid}.json")
            for method, (solver, cache, tag) in solver_by_method.items():
                result = case["methods"][method]
                final = [call for call in result["calls"] if call["tag"] == tag]
                if len(final) != 1:
                    raise ValueError(f"Expected one final answer request: {dataset}/{method}/{cid}")
                key = final[0]["key"]
                record = read_json(cache / f"{key}.json")
                request = record["request"]
                identity = {"request": request, "deployment_id": solver["deployment_id"]}
                actual_key = digest(json.dumps(identity, sort_keys=True, ensure_ascii=False))
                response = record["response"]["choices"][0]["message"]["content"]
                if record["key"] != key or actual_key != key or request["model"] != solver["model"] or response != result["response"]:
                    raise ValueError(f"Final request does not match the prediction: {dataset}/{method}/{cid}")
                messages = request["messages"]
                if not messages or any(not isinstance(message["content"], str) for message in messages):
                    raise ValueError(f"Expected text messages in final request: {dataset}/{method}/{cid}")
                write(directory / "final-inputs" / method / f"{cid}.json", {"messages": messages})
                counts[dataset] += 1
    print(f"Exported verified final model inputs: {counts}", flush=True)


def call_metrics(row):
    calls = row["calls"]
    prefill = sum(c.get("timings", {}).get("prompt_ms", 0) for c in calls) / 1000
    decode = sum(c.get("timings", {}).get("predicted_ms", 0) for c in calls) / 1000
    return {"score": int(row["judge"]), "max_score": 1, "fully_correct": bool(row["judge"]),
            "pred": row["pred"], "wall_seconds": row["seconds"],
            "service_seconds": prefill + decode, "prefill_seconds": prefill, "decode_seconds": decode,
            "prompt_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in calls),
            "completion_tokens": sum(c["usage"].get("completion_tokens", 0) for c in calls),
            "llm_calls": len(calls), "cached_calls": sum(bool(c["cached"]) for c in calls),
            "evidence_count": len(row.get("evidence_ids", [])), "truncated": row.get("truncated", False),
            "rounds": row.get("rounds", 0), "fallback": row.get("fallback", False)}


def export_structure(output, key, written, build):
    if key not in written:
        write(output / "knowledge" / f"{key}.json", build())
        written.add(key)
    return key


def add_frozen_results(output, dataset, info, detailed, context, predictions, grades=None):
    for method, row in predictions.items():
        grade = grades[method] if grades is not None else None
        metrics, details = modern_result(row, context, info["context_sha256"], info["answer"], grade)
        if grade is not None and metrics["max_score"] != info["max_score"]:
            raise ValueError(f"Frozen scoring maximum differs for {row['id']}")
        info["methods"][method] = metrics
        detailed["methods"][method] = details
        if details["has_trace"]:
            prefix = output / "math" if dataset == "math" else output
            write(prefix / "traces" / method / f"{row['id']}.json", row)


def export_math(root, output, questions, conditions, sources, structures_written):
    context_path = root / "data/MATH_2025_NATIONAL_I/textbook_chunks.jsonl"
    context, meta = stitch_context(context_path)
    sources[context_path] = file_digest(context_path)
    historical, historical_bm25 = read_historical_math(
        root, {question["id"] for question in questions}, conditions["repar_r27_08b"][4], sources
    )
    conditions = historical | conditions
    indexes = {}
    for method in INDEX_METHODS:
        info_path = root / RUNS[method][1] / "math/index_reuse.json"
        name = read_json(info_path)["path"]
        sources[info_path] = file_digest(info_path)
        index = load_frozen_index(root, name, conditions[method][4], sources)
        indexes[method] = export_structure(output, Path(name).stem, structures_written,
                                           lambda: tree_graph_data(index, context))
    bm25_stats = conditions["bm25_27b"][1][questions[0]["id"]]["index_stats"]
    indexes["bm25_27b"] = export_structure(output, "bm25-math", structures_written,
                                           lambda: bm25_data(context, bm25_stats))
    partition = bm25_data(context, historical_bm25)
    chunks = {chunk["id"]: chunk for chunk in partition["chunks"]}
    for row in conditions["bm25_08b"][1].values():
        for record in row["evidence_records"]:
            if chunks[record["id"]] != {key: record[key] for key in ("id", "start", "end", "text")}:
                raise ValueError(f"Historical BM25 chunk differs from recorded evidence: {row['id']}")
    same_partition = all(historical_bm25[key] == bm25_stats[key] for key in ("chunk_words", "leaves", "context_sha256"))
    indexes["bm25_08b"] = indexes["bm25_27b"] if same_partition else export_structure(
        output, "bm25-math-08b", structures_written, lambda: partition
    )
    kinds = {"single_choice": "单选题", "multiple_choice": "多选题",
             "fill_in_blank": "填空题", "constructed_response": "解答题"}
    listing = []
    for question in questions:
        cid = question["id"]
        public = public_question(question)
        predictions = {method: condition[1][cid] for method, condition in conditions.items()}
        grades = {method: condition[2][cid] for method, condition in conditions.items()}
        for method, row in predictions.items():
            if row["question"] != public or (method not in historical and row["task"] != task_text(question)):
                raise ValueError(f"Frozen math question mismatch: {method} {cid}")
        info = {
            "id": cid, "number": question["question_number"], "question": question["question_stem"],
            "choices": {choice["tag"]: choice["text"] for choice in question["choices"]},
            "answer": question["answer_key"], "max_score": question["score"],
            "sub_domain": kinds[question["question_type"]], "context_chars": len(context),
            "context_sha256": meta["sha256"], "methods": {},
        }
        detailed = {**info, "methods": {}, "evidence": {}, "structures": indexes,
                    "question_data": public, "reference_solution": question["reference_solution"],
                    "scoring_notes": question["scoring_notes"], "answer_values": question["answer_values"]}
        add_frozen_results(output, "math", info, detailed, context, predictions, grades)
        write(output / "math/cases" / f"{cid}.json", detailed)
        listing.append(info)
    write(output / "math/overview.json", {
        "title": "MATH 2025 · 全国 I 卷", "case_count": len(listing), "cases": listing,
        "methods": {method: item[0] for method, item in (MATH_BASELINES | RUNS).items()},
        "models": {method: condition[3] for method, condition in conditions.items()},
        "default_x": "direct_08b", "default_y": V5_METHOD,
    })


def export(root=ROOT, output=None):
    root = Path(root)
    output = Path(output or root / "webserver/data")
    metadata = read_json(root / "artifacts/experiment/run_metadata.json")
    config = metadata["config"]
    methods = dict(METHODS)
    metadata_by_method={m:metadata for m in methods}
    paths_by_method={m:root/f"artifacts/experiment/{m}.jsonl" for m in methods}
    v2_root=root/"artifacts/experiment_v2"
    if all(len(read_jsonl(v2_root/(m+".jsonl")))==125 for m in ("repar","repar_tree")):
        v2_metadata=read_json(v2_root/"run_metadata.json")
        methods["repar"]="RePAR v1"
        methods["repar_tree"]="RePAR v1 · 禁用图"
        for method,label in (("repar","RePAR v2"),("repar_tree","RePAR v2 · 无图")):
            key=method+"_v2"
            methods[key]=label
            metadata_by_method[key]=v2_metadata
            paths_by_method[key]=v2_root/(method+".jsonl")
    if config["model"] != "Qwen3.5-0.8B-Q4_K_M":
        raise ValueError("Dashboard must use the standard Q4 experiment")
    data_path = root / "data/longbench_v2.json"
    data = read_json(data_path)
    samples = [x for x in data if x["domain"] == "Multi-Document QA"]
    if len(samples) != 125:
        raise ValueError("Expected all 125 Multi-Document QA questions")
    math_path = root / "data/MATH_2025_NATIONAL_I/MATH_2025_NATIONAL_I_QA.jsonl"
    math_questions = read_jsonl(math_path)
    math_ids = {question["id"] for question in math_questions}
    if len(math_questions) != 19 or len(math_ids) != 19:
        raise ValueError("Expected all 19 unique math questions")
    new_sources = {path: file_digest(path) for path in (math_path, data_path)}
    conditions = {method: read_condition(root, method, {s["_id"] for s in samples}, math_ids, new_sources)
                  for method in RUNS}
    structures_written = set()
    export_math(root, output, math_questions, conditions, new_sources, structures_written)
    rows = {}
    source_files = [root / "artifacts/experiment/run_metadata.json", root / "artifacts/dataset_manifest.json"]
    if "repar_v2" in methods:
        source_files.append(v2_root/"run_metadata.json")
    for method in methods:
        path = paths_by_method[method]
        values = read_jsonl(path)
        if len(values) != 125 or {x["_id"] for x in values} != {x["_id"] for x in samples}:
            raise ValueError(f"Incomplete or duplicate condition: {method}")
        rows[method] = {x["_id"]: x for x in values}
        source_files.append(path)
    listing = []
    previous = Path.cwd()
    try:
        os.chdir(root)
        for number, sample in enumerate(samples, 1):
            cid = sample["_id"]
            context_hash = digest(sample["context"])
            method_rows = {m: rows[m][cid] for m in methods}
            for method,row in method_rows.items():
                assert row["config_sha256"] == metadata_by_method[method]["config_sha256"]
                assert row["context_sha256"] == context_hash
                assert row["answer"] == sample["answer"]
                assert row["judge"] == (row["pred"] == sample["answer"])
            traces = {m: read_json(root / method_rows[m]["trace_path"]) for m in methods if m.startswith("repar")}
            source_files.extend(root / method_rows[m]["trace_path"] for m in traces)
            evidence_ids = {i for row in method_rows.values() for i in row.get("evidence_ids", [])}
            retrieved = {}
            for m, trace in traces.items():
                discovered = []
                for step in trace["trace"]:
                    observations = [c.get("observation", {}) for c in step.get("executed", [])]
                    if step.get("fallback"):
                        observations.append(step.get("observation", {}))
                    for observation in observations:
                        for record in observation.get("results", []):
                            evidence_ids.add(record["id"])
                            if record["id"] not in discovered:
                                discovered.append(record["id"])
                retrieved[m] = discovered
            index = KnowledgeIndex.cached(sample["context"], config)
            entity_graph=None
            if "repar_v2" in methods:
                from repar.index_v2 import KnowledgeIndex as EntityIndex
                entity_index=EntityIndex.cached(sample["context"],v2_metadata["config"])
                assert entity_index.spans==index.spans
                entity_graph=entity_index.entity_graph
            evidence = {}
            for eid in sorted(evidence_ids, key=lambda x: int(x[1:])):
                i = index._leaf_id(eid)
                record = index.record(i)
                assert sample["context"][record["start"]:record["end"]] == record["text"]
                knowledge = index.knowledge[i]
                evidence[eid] = {**record, "claims": knowledge["claims"],
                                 "links": [{**edge, "target": f"C{edge['target']}"} for edge in index.edges[i]]}
                if entity_graph:
                    eids=entity_graph.passage_entities[i]
                    propositions=[entity_graph.propositions[int(pid[1:])] for pid in entity_graph.passage_propositions[i]]
                    rids=list(dict.fromkeys(rid for p in propositions for rid in p["relation_ids"]))
                    evidence[eid].update({"entity_ids":eids,"entities":[entity_graph.describe(e) for e in eids],
                                         "propositions":propositions,
                                         "entity_relations":[{**entity_graph.relations[int(rid[1:])],
                                           "proposition_ids":[p["id"] for p in propositions if rid in p["relation_ids"]],
                                           "passage_ids":[eid],
                                           "total_source_count":len(entity_graph.relations[int(rid[1:])]["passage_ids"])}
                                           for rid in rids]})
            info = {"id": cid, "number": number, "question": sample["question"],
                    "choices": {c: sample["choice_"+c] for c in "ABCD"}, "answer": sample["answer"],
                    "sub_domain": sample["sub_domain"], "length": sample["length"], "difficulty": sample["difficulty"],
                    "context_chars": len(sample["context"]), "context_sha256": context_hash,
                    "original_prompt_tokens": method_rows["direct"]["original_prompt_tokens"],
                    "methods": {m: call_metrics(row) for m, row in method_rows.items()}}
            listing.append(info)
            detailed = {**info, "evidence": evidence, "index_stats": index.stats,
                        "methods": {m: {**info["methods"][m], "response": row["response"],
                                        "evidence_ids": row.get("evidence_ids", []),
                                        "retrieved_ids": retrieved.get(m, row.get("evidence_ids", [])),
                                        "termination": row.get("termination"), "calls": row["calls"]}
                                    for m, row in method_rows.items()}}
            predictions = {m: condition[0][cid] for m, condition in conditions.items()}
            add_frozen_results(output, "longbench", info, detailed, sample["context"], predictions)
            structure_ids = {}
            legacy_key = export_structure(output, f"passage-{cid}", structures_written,
                                          lambda: tree_graph_data(index, sample["context"]))
            structure_ids.update({m: legacy_key for m in ("repar", "repar_tree")})
            if entity_graph:
                original_config = v2_metadata["config"]
                index_options = {k: original_config.get(k) for k in
                                 ("chunk_words", "tree_fanout", "graph_model", "index_version")}
                original_key = digest(sample["context"] + json.dumps(index_options, sort_keys=True))
                legacy_v2 = export_structure(output, original_key, structures_written,
                                             lambda: tree_graph_data(entity_index, sample["context"]))
                structure_ids.update({m: legacy_v2 for m in ("repar_v2", "repar_tree_v2")})
            for method in INDEX_METHODS:
                name = predictions[method]["index_reuse"]["path"]
                key = Path(name).stem
                if root / name not in new_sources or key not in structures_written:
                    current_index = load_frozen_index(root, name, conditions[method][4], new_sources)
                    export_structure(output, key, structures_written,
                                     lambda: tree_graph_data(current_index, sample["context"]))
                if new_sources[root / name] != conditions[method][4][name]:
                    raise ValueError(f"Conflicting frozen sources for {cid}: {name}")
                structure_ids[method] = key
            bm25_stats = predictions["bm25_27b"]["index_stats"]
            bm25_key = export_structure(output, f"bm25-{cid}", structures_written,
                                        lambda: bm25_data(sample["context"], bm25_stats))
            structure_ids.update({"bm25": bm25_key, "bm25_27b": bm25_key})
            detailed["structures"] = structure_ids
            for method,row in method_rows.items():
                if method.endswith("_v2"):
                    attempts=[]
                    for call in row["calls"]:
                        if not any(call["tag"].startswith("v2_"+phase+"_") for phase in ("plan","action","reflection")):
                            continue
                        raw=read_json(root/"artifacts/cache/llm"/(call["key"]+".json"))
                        attempts.append({"tag":call["tag"],"finish_reason":call.get("finish_reason"),
                                         "response":raw["response"]["choices"][0]["message"].get("content", ""),
                                         "cached":call["cached"],"usage":call["usage"]})
                    detailed["methods"][method]["phase_attempts"]=attempts
            write(output / "cases" / (cid+".json"), detailed)
            for method, trace in traces.items():
                assert trace["context_sha256"] == context_hash
                write(output / "traces" / method / (cid+".json"), trace)
            print(f"Export {number}/125 {cid}", flush=True)
    finally:
        os.chdir(previous)
    summary = read_json(root / "artifacts/experiment/summary.json")
    overall={m:summary["methods"][m]["overall"] for m in METHODS}
    if "repar_v2" in methods:
        v2_summary=read_json(v2_root/"summary.json")
        overall.update({m:v2_summary["methods"][m]["overall"] for m in ("repar_v2","repar_tree_v2")})
    for method, (label, _) in RUNS.items():
        methods[method] = label
        overall[method] = {"accuracy": 100 * sum(item["methods"][method]["score"] for item in listing) / len(listing)}
    write(output / "overview.json", {
        "title": "LongBench v2 · Multi-Document QA", "model": config["model"], "methods": methods,
        "case_count": len(listing), "cases": listing,
        "overall": overall, "default_y": V5_METHOD,
        "models": {**{m: value["config"]["model"] for m, value in metadata_by_method.items()},
                   **{m: value[3] for m, value in conditions.items()}},
        "indexing": summary["indexing"], "structure_audit": summary["structure_audit"],
        "timing_note": "在线时间含索引加载、检索、LLM 和本地处理；首次建库单列。模型计算时间为各次真实调用的预填充与解码之和，缓存调用仍计入原始计算成本。",
        "config_sha256": metadata["config_sha256"],
    })
    write(output / "manifest.json", {
        "conditions": list(methods), "cases": 125, "predictions": 125*len(methods), "model": config["model"],
        "datasets": {"longbench": {"cases": 125, "conditions": list(methods)},
                     "math": {"cases": 19, "conditions": list(MATH_BASELINES | RUNS)}},
        "sources": {p.relative_to(root).as_posix(): sha for p, sha in
                    ({p: file_digest(p) for p in source_files} | new_sources).items()},
        "evidence_verified_against_source": True,
    })
    export_final_inputs(root, output)
    print(f"Exported 125 LongBench and 19 math cases to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-inputs-only", action="store_true", help="Export final requests for existing dashboard cases")
    if parser.parse_args().final_inputs_only:
        export_final_inputs()
    else:
        export()
