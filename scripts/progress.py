"""Operational progress only, deliberately no accuracy inspection until all conditions finish."""
import json
import argparse
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", default="artifacts/experiment")
args = p.parse_args()
root = Path(args.root)
for method in ("direct", "bm25", "repar", "repar_tree"):
    p = root / (method + ".jsonl")
    rows = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    tokens = sum(sum(c.get("usage", {}).get("total_tokens", 0) for c in r.get("calls", [])) for r in rows)
    print(f"{method:12} {len(rows):3}/125  recorded_tokens={tokens:,}")
metadata = root / "run_metadata.json"
trace_dir = json.loads(metadata.read_text(encoding="utf-8"))["config"]["trace_dir"] if metadata.exists() else "artifacts/traces_qwen08_standard"
traces = list((Path(trace_dir) / "repar").glob("*.json"))
if traces:
    last = max(traces, key=lambda x: x.stat().st_mtime)
    trace = json.loads(last.read_text(encoding="utf-8"))
    actions = [c.get("name") for t in trace["trace"] for c in t.get("executed", []) if c.get("name")]
    print("Last RePAR trace:", last.stem, "actions:", actions, "termination:", trace["termination"])
errors = root / "errors.jsonl"
if errors.exists():
    print("Recorded errors:", len(errors.read_text(encoding="utf-8").splitlines()))
