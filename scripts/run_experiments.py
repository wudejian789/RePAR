"""Run/resume every prespecified 125-question condition on one local GPU.

No scores are inspected here. Keep this process detached from the terminal so a
client reconnect does not interrupt a multi-hour experiment.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

from repar.io import read_json, read_jsonl, write_json


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
VARIANTS = [
    ("standard", "configs/qwen08_standard.json", "artifacts/experiment", []),
    ("mtp_off", "configs/qwen08_mtp_off.json", "artifacts/experiment_mtp_off", ["-MtpWeightsOnly"]),
    ("mtp_on", "configs/qwen08_mtp.json", "artifacts/experiment_mtp", ["-Mtp"]),
]
METHODS = ["repar", "bm25", "repar_tree", "direct"]


def command(args, log):
    print(time.strftime("%FT%T%z"), "RUN", " ".join(args), flush=True)
    with Path(log).open("a", encoding="utf-8") as f:
        subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def state_path(variant):
    p = Path("artifacts/deployment") / variant / "server_process.json"
    return p if p.exists() else Path("artifacts/deployment/server_process.json")


def stop_server(variant):
    p = state_path(variant)
    if p.exists():
        state = json.loads(p.read_text(encoding="utf-8-sig"))
        # Only stop the PID saved by our own launcher, after verifying its image.
        cmd = (f"$p=Get-Process -Id {int(state['pid'])} -ErrorAction SilentlyContinue; "
               "if ($p -and $p.ProcessName -eq 'llama-server') { Stop-Process -InputObject $p }")
        subprocess.run(["powershell", "-NoProfile", "-Command", cmd], check=True)


def ready(config, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(config["base_url"] + "/health", timeout=3)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("Local llama.cpp server did not become healthy")


def main():
    protocol = {
        "fixed_before_aggregate_scoring": True,
        "variants": [{"variant": v, "config": c, "output": o} for v, c, o, _ in VARIANTS],
        "methods": METHODS, "samples_per_condition": 125,
        "total_benchmark_predictions": 1500,
        "mtp_default_criterion": {
            "accuracy": "MTP on must be no more than 1 question lower than BOTH standard and MTP off on each of direct and RePAR; report paired intervals, not equivalence proof",
            "speed": "MTP on must have at least 20% lower mean summed server prefill+decode time for RePAR and at least 20% higher same-input synthetic decode throughput than BOTH standard and MTP off",
            "scope": "Operational rule only; 125 examples cannot establish a narrow statistical equivalence margin"},
        "no_label_based_tuning": True,
    }
    if not Path("artifacts/experiment_protocol.json").exists():
        write_json("artifacts/experiment_protocol.json", protocol)
    previous = None
    for variant, config_path, output, flags in VARIANTS:
        config = read_json(config_path)
        complete = all(len(read_jsonl(Path(output) / (m + ".jsonl"))) == 125 for m in METHODS)
        if complete:
            print("ALREADY COMPLETE", variant, flush=True)
            previous = variant
            continue
        try:
            health = requests.get(config["base_url"] + "/health", timeout=2).status_code == 200
        except requests.RequestException:
            health = False
        # A resumed variant may already have its own server running.
        current_state = state_path(variant)
        state = json.loads(current_state.read_text(encoding="utf-8-sig")) if current_state.exists() else {}
        args = state.get("arguments", [])
        same_variant = ((variant == "standard" and "MTP" not in str(state.get("model", ""))) or
                        (variant == "mtp_on" and "draft-mtp" in args) or
                        (variant == "mtp_off" and "MTP" in str(state.get("model", "")) and "draft-mtp" not in args))
        if not health or not same_variant:
            if health:
                stop_server(previous or "standard")
            command(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                     "scripts/serve_local.ps1", *flags], f"artifacts/{variant}_launch.log")
        ready(config)
        props = requests.get(config["base_url"] + "/props", timeout=20).json()
        write_json(f"artifacts/deployment/{variant}/server_metadata.json", props)
        command([PYTHON, "-X", "utf8", "scripts/mtp_probe.py", "--config", config_path,
                 "--output", f"artifacts/deployment/probe_{variant}.json"], f"artifacts/{variant}_probe.log")
        for method in METHODS:
            command([PYTHON, "-X", "utf8", "-m", "repar.cli", "run", "--method", method,
                     "--config", config_path, "--output", output], f"artifacts/{variant}_suite.log")
            assert len(read_jsonl(Path(output) / (method + ".jsonl"))) == 125
        if variant == "standard":
            for name in ("server.stdout.log", "server.stderr.log", "server_process.json"):
                source = Path("artifacts/deployment") / name
                if source.exists():
                    target = Path("artifacts/deployment/standard") / name
                    target.write_bytes(source.read_bytes())
        print("COMPLETED", variant, flush=True)
        previous = variant
    write_json("artifacts/all_conditions_complete.json", {"completed": time.strftime("%FT%T%z"),
                                                            "conditions": 12, "predictions": 1500})
    print("ALL 1500 PREDICTIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
