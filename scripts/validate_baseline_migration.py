"""Offline regression audit against every recorded standard-Q4 BM25 request.

Uses archived requests/responses as fixtures; makes no network or model calls and
does not overwrite predictions, benchmark data or historical manifests.
"""
from pathlib import Path
import argparse
import tempfile

from baselines.BM25_RAG.adapters.longbench import answer
from repar.client import LLMClient
from repar.io import load_public_samples, public_item, read_json, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/longbench_v2.json")
    parser.add_argument("--results", default="artifacts/experiment")
    parser.add_argument("--output", default="artifacts/maintenance/bm25_migration_validation.json")
    args = parser.parse_args()
    root = Path(args.results)
    config = read_json(root/"run_metadata.json")["config"]
    previous = read_jsonl(root/"bm25.jsonl")
    samples = {sample["_id"]: sample for sample in load_public_samples(args.data)}
    assert len(previous) == len(samples) == len({r["_id"] for r in previous})
    assert set(samples) == {r["_id"] for r in previous}
    checked = []
    with tempfile.TemporaryDirectory(prefix="repar-bm25-migration-") as temp:
        for row in previous:
            saved = read_json(Path("artifacts/cache/llm")/(row["calls"][0]["key"]+".json"))
            client = LLMClient(config, cache_dir=Path(temp)/row["_id"])

            def recorded_response(route, body):
                assert route == "/v1/chat/completions"
                assert body == saved["request"], f"Model input changed for {row['_id']}"
                return saved["response"]

            client.post = recorded_response
            result = answer(public_item(samples[row["_id"]]), None, client, config)
            assert result["evidence_ids"] == row["evidence_ids"]
            assert result["pred"] == row["pred"] and result["response"] == row["response"]
            assert result["context_sha256"] == row["context_sha256"]
            assert client.calls[0]["key"] == row["calls"][0]["key"]
            checked.append(row["_id"])
    write_json(args.output, {"all_checks_passed": True, "cases": len(checked), "ids": checked,
                             "source_results": str(root/"bm25.jsonl"),
                             "checks": ["same selected evidence IDs", "byte-equivalent model request JSON values",
                                        "same request identity", "same recorded answer and source hash"],
                             "model_calls": 0, "historical_records_changed": False,
                             "timing_note": "No inference speed or fresh accuracy claim; standalone build statistics now measure BM25 only."})
    print(f"Verified all {len(checked)} BM25 cases: evidence, model requests and recorded predictions unchanged; no model calls.")


if __name__ == "__main__":
    main()
