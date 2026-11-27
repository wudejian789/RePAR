"""Adapt frozen evaluation records to the Case Lab's display schema."""
from __future__ import annotations

import hashlib
import json

from repar.io import digest, read_json


V5_METHOD = "repar_v5_08b"
RUNS = {
    "direct_27b": ("直接回答 · 27B", "artifacts/baselines_27b_20261119/direct"),
    "bm25_27b": ("BM25 RAG · 27B", "artifacts/baselines_27b_20261119/bm25"),
    "repar_r27_08b": ("RePAR v4-r27 · 0.8B", "artifacts/repar_v4_r27_20261120"),
    "repar_r27_27b": ("RePAR v4-r27 · 27B", "artifacts/repar_v4_r27_27b_20261120"),
    V5_METHOD: ("RePAR v5 · 0.8B · seed 42", "artifacts/repar_v5_20261126/seed_42"),
}
INDEX_METHODS = ("repar_r27_08b", "repar_r27_27b", V5_METHOD)
HISTORICAL_MATH_RUN = "artifacts/math_2025_national_i_v5"
MATH_BASELINES = {
    "direct_08b": ("直接回答 · 0.8B", "direct"),
    "bm25_08b": ("BM25 RAG · 0.8B", "bm25"),
}


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_rows(path, expected_hash, ids, sources):
    actual_hash = file_digest(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(f"Frozen output hash mismatch: {path}")
    sources[path] = actual_hash
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"Duplicate ID {row['id']}: {path}")
            rows[row["id"]] = row
    if rows.keys() != ids:
        raise ValueError(f"Unexpected question coverage: {path}")
    return rows


def read_condition(root, method, ids, math_ids, sources):
    directory = root / RUNS[method][1]
    if method == V5_METHOD:
        return read_v5_condition(directory, ids, math_ids, sources)
    if method.startswith("repar"):
        lock_path = directory / "completion.json"
        lock = read_json(lock_path)
        hashes = lock["files_sha256"]
    else:
        lock_path = directory.parent / "completion.json"
        lock = read_json(lock_path)
        hashes = lock["method_provenance"][directory.name]["input_sha256"]
    if not lock["complete"]:
        raise ValueError(f"Incomplete run: {directory}")
    sources[lock_path] = file_digest(lock_path)
    protocol_path = directory / "protocol.json"
    expected = lock.get("frozen_generation", {}).get("protocol_sha256") or hashes["protocol.json"]
    actual = file_digest(protocol_path)
    if actual != expected:
        raise ValueError(f"Frozen protocol hash mismatch: {protocol_path}")
    sources[protocol_path] = actual
    protocol = read_json(protocol_path)
    solver = protocol["identity"]["solver"]
    model = solver["model"] if "model" in solver else solver["config"]["model"]
    longbench = checked_rows(directory / "longbench/predictions.jsonl",
                             hashes["longbench/predictions.jsonl"], ids, sources)
    math = checked_rows(directory / "math/predictions.jsonl",
                        hashes["math/predictions.jsonl"], math_ids, sources)
    grades = checked_rows(directory / "math/grades.jsonl", hashes["math/grades.jsonl"], math_ids, sources)
    index_references = protocol["identity"].get("references", {})
    return longbench, math, grades, model, index_references


def read_v5_condition(directory, ids, math_ids, sources):
    audit_path = directory / "audit_v5.json"
    audit = read_json(audit_path)
    if not audit["passed"]:
        raise ValueError(f"Incomplete audit: {directory}")
    protocol_path = directory / "protocol.json"
    identity = read_json(protocol_path)["identity"]
    sources.update({path: file_digest(path) for path in (audit_path, protocol_path)})
    predictions = {}
    for name, case_ids in (("longbench", ids), ("math", math_ids)):
        manifest_path = directory / name / "generation_manifest.json"
        manifest = read_json(manifest_path)
        sources[manifest_path] = file_digest(manifest_path)
        rows = checked_rows(manifest_path.with_name("predictions.jsonl"),
                            manifest["predictions_sha256"], case_ids, sources)
        if any(row["agent_version"] != "react-v5" or row["status"] != "ok" for row in rows.values()):
            raise ValueError(f"Unexpected prediction in audited v5 condition: {name}")
        predictions[name] = rows
    grade_path = directory / "math/grades.jsonl"
    grades = checked_rows(grade_path, None, math_ids, sources)
    if (abs(sum(grade["score"] for grade in grades.values()) - audit["stats"]["math"]["score"]) > 1e-9
            or sum(row["correct"] for row in predictions["longbench"].values()) != audit["stats"]["longbench"]["score"]):
        raise ValueError(f"Export scores differ from the v5 audit: {directory}")
    return predictions["longbench"], predictions["math"], grades, identity["solver"]["model"], identity["references"]


def read_historical_math(root, ids, references, sources):
    directory = root / HISTORICAL_MATH_RUN
    metadata = {}
    for name in ("completion", "protocol", "generation_manifest", "bm25_build"):
        path = directory / f"{name}.json"
        metadata[name] = read_json(path)
        sources[path] = file_digest(path)
    if not metadata["completion"]["audit_passed"]:
        raise ValueError(f"Historical math audit did not pass: {directory}")
    solver = metadata["protocol"]["identity"]["resolved_solver"]
    stats = metadata["bm25_build"]["index_stats"]
    if stats["chunk_words"] != solver["chunk_words"]:
        raise ValueError(f"Historical BM25 partition differs from its protocol: {directory}")

    conditions = {}
    for method, (_, name) in MATH_BASELINES.items():
        prediction_name = f"{name}.jsonl"
        grade_name = f"{name}_grades.jsonl"
        prediction_hash = metadata["generation_manifest"]["prediction_sha256"][prediction_name]
        if prediction_hash != references[f"{HISTORICAL_MATH_RUN}/{prediction_name}"]:
            raise ValueError(f"Conflicting historical math hashes: {prediction_name}")
        predictions = checked_rows(directory / prediction_name, prediction_hash, ids, sources)
        grades = checked_rows(directory / grade_name, references[f"{HISTORICAL_MATH_RUN}/{grade_name}"], ids, sources)
        if any(row["method"] != name for row in [*predictions.values(), *grades.values()]):
            raise ValueError(f"Unexpected historical math method: {name}")
        conditions[method] = (None, predictions, grades, solver["model"], {})
    return conditions, stats


def verified_record(record, context):
    start, end = record["start"], record["end"]
    if not 0 <= start <= end <= len(context) or context[start:end] != record["text"]:
        raise ValueError(f"Evidence {record['id']} differs from its source [{start}:{end}]")
    return record


def modern_result(row, context, context_hash, answer, grade=None):
    if row["context_sha256"] != context_hash:
        raise ValueError(f"Context mismatch: {row['id']}")
    if grade is None:
        if row["gold"] != answer or row["correct"] != (row["pred"] == answer):
            raise ValueError(f"Frozen match score mismatch: {row['id']}")
        score, full, pred, maximum = int(row["correct"]), row["correct"], row["pred"], 1
    else:
        if (digest(row["response"]) != grade["response_sha256"]
                or not 0 <= grade["score"] <= grade["max_score"]):
            raise ValueError(f"Invalid grade for {row['id']}")
        score, full, maximum = (grade[k] for k in ("score", "fully_correct", "max_score"))
        pred = grade.get("prediction")
    calls = row["calls"]
    replies = [call["response"] for call in calls]
    prefill = sum(reply.get("timings", {}).get("prompt_ms", 0) for reply in replies) / 1000
    decode = sum(reply.get("timings", {}).get("predicted_ms", 0) for reply in replies) / 1000
    windows = [verified_record(record, context) for record in row.get("evidence_records", [])]
    records = {record["id"]: record for record in windows}
    if list(records) != row["evidence_ids"]:
        raise ValueError(f"Final evidence order differs from frozen record: {row['id']}")
    retrieved = {}
    for step in row["trace"]:
        for call in step["executed"]:
            for record in call.get("observation", {}).get("results", []):
                key = (record["id"], record["start"], record["end"])
                retrieved[key] = verified_record(record, context)
            for view in call.get("visible_observation", {}).get("source_views", []):
                verified_record(view, context)
    metrics = {
        "score": score, "max_score": maximum, "fully_correct": full, "pred": pred,
        "wall_seconds": row["seconds"], "service_seconds": prefill + decode,
        "prefill_seconds": prefill, "decode_seconds": decode,
        "prompt_tokens": sum(reply.get("usage", {}).get("prompt_tokens", 0) for reply in replies),
        "completion_tokens": sum(reply.get("usage", {}).get("completion_tokens", 0) for reply in replies),
        "final_prompt_tokens": row["rendered_input_tokens"],
        "llm_calls": len(calls), "cached_calls": sum(bool(call["cached"]) for call in calls),
        "evidence_count": len(windows), "rounds": len(row["trace"]),
        "truncated": row.get("truncated", row.get("direct", {}).get("truncated", False)),
    }
    retrieved_by_id = {record["id"]: record for record in retrieved.values()}
    details = {
        **metrics, "response": row["response"], "evidence_ids": list(records),
        "retrieved_ids": list(retrieved_by_id | records), "evidence": retrieved_by_id | records,
        "evidence_records": windows, "retrieved_records": list(retrieved.values()),
        "termination": row.get("termination"), "calls": calls, "grade": grade,
        "has_trace": row.get("agent_version", "").startswith("react-"),
        "finish_reason": replies[-1]["choices"][0]["finish_reason"] if replies else None,
    }
    return metrics, details
