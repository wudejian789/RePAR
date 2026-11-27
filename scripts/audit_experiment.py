"""Verify every scored row against data, frozen configuration and raw API cache."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from repar.evaluation import extract_answer
from repar.io import digest, load_public_samples, read_json, read_jsonl, write_json


data = {r["_id"]: r for r in load_public_samples("data/longbench_v2.json")}
frozen = read_json("artifacts/implementation_manifest.json")
default_changes = []
for filename, expected in frozen["files"].items():
    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if actual != expected and Path(filename).as_posix() == "configs/experiment.json":
        # Only the deployment default may change AFTER all inference is done.
        # The three condition-specific configurations must still match the freeze.
        selected = read_json("artifacts/default_selection.json")
        assert selected["after_all_predictions_complete"]
        assert actual == selected["selected_file_sha256"]
        assert expected == selected["previous_default_sha256"]
        assert hashlib.sha256(Path(selected["selected_config"]).read_bytes()).hexdigest() == actual
        default_changes.append(selected)
    else:
        assert actual == expected, f"Frozen file changed: {filename}"
conditions = []
deployments = {}
for variant, expected_spec in (("standard", "none"), ("mtp_off", "none"), ("mtp_on", "draft-mtp")):
    props = read_json(f"artifacts/deployment/{variant}/server_metadata.json")
    state = json.loads(Path(f"artifacts/deployment/{variant}/server_process.json").read_text(encoding="utf-8-sig"))
    observed = "draft-mtp" if "draft-mtp" in state["arguments"] else "none"
    assert observed == expected_spec, (variant, observed)
    # In b12500 /props describes per-request defaults and still prints "none"
    # for CLI-initialized MTP. Verify actual draft work as well as launch flags.
    probe = read_json(f"artifacts/deployment/probe_{variant}.json")
    drafts = sum(r["timings"].get("draft_n", 0) for r in probe["rows"])
    assert (drafts > 0) == (variant == "mtp_on")
    log = Path(f"artifacts/deployment/{variant}/server.stderr.log").read_text(encoding="utf-8", errors="replace")
    assert ("creating MTP draft context" in log) == (variant == "mtp_on")
    deployments[variant] = {"speculative_types_from_cli": observed,
                            "props_request_default": props["default_generation_settings"]["params"].get("speculative.types"),
                            "probe_draft_tokens_verified": drafts,
                            "build": props["build_info"], "model": props["model_alias"]}
for root in ("artifacts/experiment", "artifacts/experiment_mtp_off", "artifacts/experiment_mtp"):
    metadata = read_json(Path(root)/"run_metadata.json")
    config = metadata["config"]
    assert digest(json.dumps(config, sort_keys=True)) == metadata["config_sha256"]
    for method in ("direct", "bm25", "repar", "repar_tree"):
        rows = read_jsonl(Path(root)/(method+".jsonl"))
        assert len(rows) == len(data) == len({r["_id"] for r in rows})
        assert {r["_id"] for r in rows} == set(data)
        calls, cache_hits = 0, 0
        for row in rows:
            sample = data[row["_id"]]
            assert row["config_sha256"] == metadata["config_sha256"]
            assert row["context_sha256"] == digest(sample["context"])
            assert row["answer"] == sample["answer"]
            assert row["pred"] == extract_answer(row["response"])
            assert row["judge"] == (row["pred"] == sample["answer"])
            for call in row["calls"]:
                raw = read_json(Path("artifacts/cache/llm")/(call["key"]+".json"))
                identity = {"request": raw["request"], "deployment_id": config["deployment_id"]}
                assert digest(json.dumps(identity, sort_keys=True, ensure_ascii=False)) == call["key"]
                assert raw["request"]["model"] == config["model"]
                assert raw["response"].get("usage", {}) == call["usage"]
                assert raw["response"].get("timings", {}) == call["timings"]
                calls += 1
                cache_hits += bool(call["cached"])
            assert raw["response"]["choices"][0]["message"]["content"] == row["response"]
            if method.startswith("repar"):
                trace = read_json(row["trace_path"])
                assert trace["_id"] == row["_id"] and trace["response"] == row["response"]
                assert trace["context_sha256"] == row["context_sha256"]
                assert "answer" not in trace
        conditions.append({"root": root, "method": method, "samples": len(rows),
                           "raw_api_records_verified": calls, "cached_call_references": cache_hits})
manifests = {}
for name in ("model_manifest", "mtp_model_manifest"):
    manifest = read_json(f"artifacts/deployment/{name}.json")
    folder = "Qwen3.5-0.8B" if name == "model_manifest" else "Qwen3.5-0.8B-MTP"
    model = Path("models")/folder/manifest["filename"]
    actual = hashlib.file_digest(model.open("rb"), "sha256").hexdigest()
    assert actual == manifest["sha256"]
    manifests[name] = actual
write_json("artifacts/experiment_audit.json", {
    "time": time.strftime("%FT%T%z"), "all_checks_passed": True,
    "total_predictions": sum(c["samples"] for c in conditions), "conditions": conditions,
    "frozen_files_verified": len(frozen["files"]), "model_sha256_verified": manifests,
    "post_experiment_default_changes": default_changes,
    "deployments_verified": deployments,
    "limitations": "Verifies recorded provenance, consistency and cache identity; does not prove model truthfulness or data contamination absence."})
print("Verified all 1500 predictions, frozen files, model hashes and every API record.")
