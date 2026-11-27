"""Freeze both backbone conditions before executing either full condition."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from repar.benchmarks.v3_suite import freeze
from repar.io import read_json, write_json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/backbone_27b_comparison.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = read_json(root/args.config)
    output = root/config["output"]
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output/"protocol.json"
    if not protocol_path.exists():
        conditions = []
        for name in config["conditions"]:
            condition, target = freeze(name)
            conditions.append({"config": name, "output": condition["output"],
                "protocol_sha256": sha(target/"protocol.json")})
        identity = {"config": config, "config_sha256": sha(root/args.config),
            "conditions": conditions, "historical_reference_sha256": {p: sha(root/p) for p in config["historical_references"]}}
        write_json(protocol_path, {"frozen_at": datetime.now().astimezone().isoformat(), "identity": identity})
    protocol = read_json(protocol_path)
    if protocol["identity"]["config"] != config:
        raise ValueError("Paired protocol changed; create a new experiment")
    for condition in protocol["identity"]["conditions"]:
        target = root/condition["output"]
        if sha(target/"protocol.json") != condition["protocol_sha256"]:
            raise ValueError("Frozen condition changed")
        print("START", condition["config"], flush=True)
        with (target/"run.log").open("a", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-X", "utf8", "-u", "scripts/run_repar_v3.py",
                "--config", condition["config"], "--phase", "all"], cwd=root, stdout=stream,
                stderr=subprocess.STDOUT, check=True)
        print("COMPLETE", condition["config"], flush=True)
    write_json(output/"generation_complete.json", {"completed_at": datetime.now().astimezone().isoformat(),
        "conditions": protocol["identity"]["conditions"], "status": "generation_and_scoring_complete_audit_pending"})


if __name__ == "__main__": main()
