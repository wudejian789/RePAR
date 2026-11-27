"""Finish audit/report/deployment selection only after all 1500 predictions exist."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

from repar.io import read_json, write_json


p = argparse.ArgumentParser()
p.add_argument("--wait", action="store_true")
args = p.parse_args()
marker = Path("artifacts/all_conditions_complete.json")
while args.wait and not marker.exists():
    time.sleep(5)
assert marker.exists(), "Full benchmark has not completed"
subprocess.run([sys.executable, "-X", "utf8", "scripts/audit_experiment.py"], check=True)
subprocess.run([sys.executable, "-X", "utf8", "scripts/report_all.py"], check=True)
summary = read_json("artifacts/combined/summary.json")
enable = summary["mtp_decision"]["enable_mtp"]
source = Path("configs/qwen08_mtp.json" if enable else "configs/qwen08_standard.json")
default = Path("configs/experiment.json")
frozen = read_json("artifacts/implementation_manifest.json")
original = next(h for name, h in frozen["files"].items() if Path(name).as_posix() == "configs/experiment.json")
selection = {"selected_config": str(source), "mtp_enabled": enable,
             "selected_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "previous_default_sha256": original, "after_all_predictions_complete": True,
             "selected_at": time.strftime("%FT%T%z"), "decision": summary["mtp_decision"]}
write_json("artifacts/default_selection.json", selection)
default.write_bytes(source.read_bytes())
# The experiment ends with MTP on. If the decision rejects it, restore standard
# deployment using separate logs so the benchmark's deployment evidence survives.
try:
    current = requests.get("http://127.0.0.1:9091/props", timeout=3).json()
    already_selected = (current.get("model_alias") == read_json(source)["model"] and
                        current["default_generation_settings"]["params"]["speculative.types"] ==
                        ("draft-mtp" if enable else "none"))
except (requests.RequestException, KeyError, ValueError):
    already_selected = False
if not enable and not already_selected:
    state = json.loads(Path("artifacts/deployment/mtp_on/server_process.json").read_text(encoding="utf-8-sig"))
    stop = (f"$p=Get-Process -Id {int(state['pid'])} -ErrorAction SilentlyContinue; "
            "if ($p -and $p.ProcessName -eq 'llama-server') { Stop-Process -InputObject $p }")
    subprocess.run(["powershell", "-NoProfile", "-Command", stop], check=True)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    "scripts/serve_local.ps1", "-LogLabel", "selected_default"], check=True)
deadline = time.monotonic()+120
while time.monotonic() < deadline:
    try:
        r = requests.get("http://127.0.0.1:9091/health", timeout=3)
        if r.status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(2)
else:
    raise RuntimeError("Selected default deployment did not become healthy")
props = requests.get("http://127.0.0.1:9091/props", timeout=20).json()
assert props["default_generation_settings"]["params"]["speculative.types"] == ("draft-mtp" if enable else "none")
write_json("artifacts/deployment/selected_default_metadata.json", props)
subprocess.run([sys.executable, "-X", "utf8", "scripts/audit_experiment.py"], check=True)
files = [*Path("src").rglob("*.py"), *Path("scripts").glob("*.py"), *Path("scripts").glob("*.ps1"),
         *Path("configs").glob("*.json"), *Path("docs").glob("*.md"), Path("README.md"), Path("requirements.lock.txt")]
write_json("artifacts/delivery_manifest.json", {
    "time": time.strftime("%FT%T%z"),
    "files": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    "inference_freeze": "artifacts/implementation_manifest.json",
    "all_predictions_audited": "artifacts/experiment_audit.json",
    "default_selection": "artifacts/default_selection.json"})
write_json("artifacts/delivery_complete.json", {"completed": time.strftime("%FT%T%z"), "predictions": 1500,
                                                "mtp_enabled": enable, "default_model": props["model_alias"]})
print("DELIVERY COMPLETE", json.dumps(selection, ensure_ascii=False), flush=True)
