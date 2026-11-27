"""Resume a frozen math suite even when unrelated workspace files are added.

The algorithm/config snapshot and original protocol are never rewritten. A small
temporary workspace imports only frozen sources and links the original resources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="artifacts/math_2025_national_i_v5")
    parser.add_argument("--phase", choices=("prepare", "generate", "grade", "report", "all"), default="report")
    parser.add_argument("--method", choices=("direct", "bm25", "repar"))
    parser.add_argument("--audit", action="store_true", help="Run the separate audit with the frozen imports after this phase")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    experiment = (repository / args.experiment).resolve()
    protocol = json.loads((experiment / "protocol.json").read_text(encoding="utf-8"))
    config = protocol["identity"]["config"]
    if (repository / config["output"]).resolve() != experiment:
        raise ValueError("Experiment path must match its frozen output configuration")
    runtime = Path(tempfile.mkdtemp(prefix="repar_math_frozen_"))
    for name, expected in protocol["identity"]["source_hashes"].items():
        source = (experiment / "source" / name).resolve()
        source.relative_to((experiment / "source").resolve())
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot hash mismatch: {name}")
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Only the frozen BM25 package is needed; its parent has no workflow logic.
    (runtime / "baselines" / "__init__.py").write_text('"""Frozen baseline namespace."""\n', encoding="utf-8")
    for name in ("data", "artifacts"):
        link, target = runtime / name, (repository / name).resolve()
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           check=True, stdout=subprocess.DEVNULL)
        else:
            link.symlink_to(target, target_is_directory=True)
    config_paths = [name for name in protocol["identity"]["source_hashes"]
                    if name.startswith("configs/") and json.loads((runtime / name).read_text(encoding="utf-8")).get("experiment_id") == config["experiment_id"]]
    if len(config_paths) != 1:
        raise ValueError("Expected one frozen suite configuration")
    command = [sys.executable, "-X", "utf8", str(runtime / "scripts/evaluate_math.py"),
               "--config", config_paths[0], "--phase", args.phase]
    if args.method:
        command += ["--method", args.method]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(runtime / "src"), str(runtime)])
    event = {"time": datetime.now().astimezone().isoformat(), "phase": args.phase,
             "runtime": str(runtime), "protocol_sha256": hashlib.sha256((experiment / "protocol.json").read_bytes()).hexdigest(),
             "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "note": "Snapshot sources verified and copied; data/artifacts linked; no protocol or prediction rewriting"}
    result = subprocess.run(command, cwd=runtime, env=environment)
    event["exit_code"] = result.returncode
    if args.audit and result.returncode == 0:
        auditor = repository / "artifacts/maintenance/audit_math_benchmark.py"
        event["auditor_sha256"] = hashlib.sha256(auditor.read_bytes()).hexdigest()
        audit = subprocess.run([sys.executable, "-X", "utf8", str(auditor), "--config", config_paths[0]], cwd=runtime, env=environment)
        event["audit_exit_code"] = audit.returncode
        if audit.returncode:
            result = audit
    with (experiment / "snapshot_runs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    # Retain the tiny workspace. Recursive deletion of Windows junctions is avoided.
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
