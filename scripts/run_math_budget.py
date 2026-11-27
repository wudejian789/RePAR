"""Run one budget-only intervention using the parent's exact frozen algorithms."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolved_solver(config, base):
    result = {**base, **config["solver_overrides"]}
    result.pop("answer_grammar", None)
    result.pop("controller_schema", None)
    return result


def validate_intervention(config, parent, base):
    old, new = parent["identity"]["resolved_solver"], resolved_solver(config, base)
    diff = {k: {"old": old.get(k), "new": new.get(k)} for k in old.keys() | new.keys()
            if old.get(k) != new.get(k)}
    expected = set(config["budget_comparison"]["changed_solver_keys"])
    if set(diff) - {"trace_dir"} != expected or any(new[k] <= old[k] for k in expected):
        raise ValueError(f"Intervention must only increase declared budgets: {diff}")
    previous = parent["identity"]["config"]
    if any(config[k] != previous[k] for k in ("questions", "textbook", "judge", "methods", "solver_config")):
        raise ValueError("Data, models, judge and method list must match parent")
    return diff


def checked_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target):
            raise ValueError(f"Existing derived input differs: {target}")
    else:
        shutil.copyfile(source, target)


def prepare(root, config_path):
    config = read(root / config_path)
    parent_dir = root / config["budget_comparison"]["parent"]
    parent = read(parent_dir / "protocol.json")
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    base = read(parent_dir / "source" / config["solver_config"])
    diff = validate_intervention(config, parent, base)
    identity = {"config_sha256": sha(root / config_path), "parent_protocol_sha256": sha(parent_dir / "protocol.json"),
                "intervention": diff, "parent": config["budget_comparison"]["parent"],
                "source_hashes": parent["identity"]["source_hashes"], "notes": config["budget_comparison"]}
    lineage_path = output / "budget_protocol.json"
    if lineage_path.exists() and read(lineage_path)["identity"] != identity:
        raise ValueError("Budget comparison identity changed; choose a new output")
    runtime = Path(tempfile.mkdtemp(prefix="repar_math_budget_"))
    previous_config = next(name for name in parent["identity"]["source_hashes"] if name.startswith("configs/")
                           and read(parent_dir / "source" / name).get("experiment_id") == parent["identity"]["config"]["experiment_id"])
    for name, expected in parent["identity"]["source_hashes"].items():
        source = parent_dir / "source" / name
        if sha(source) != expected:
            raise ValueError(f"Parent frozen source changed: {name}")
        if name != previous_config:
            checked_copy(source, runtime / name)
    checked_copy(root / config_path, runtime / config_path)
    (runtime / "baselines/__init__.py").write_text('"""Frozen baseline namespace."""\n', encoding="utf-8")
    for name in ("data", "artifacts"):
        target, link = (root / name).resolve(), runtime / name
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, stdout=subprocess.DEVNULL)
        else:
            link.symlink_to(target, target_is_directory=True)
    reused, cache_keys = [], {"solver": set(), "judge": set()}
    for method in config["budget_comparison"]["reused_methods"]:
        for suffix in (".jsonl", "_grades.jsonl"):
            source, target = parent_dir / (method + suffix), output / (method + suffix)
            checked_copy(source, target)
            reused.append({"source": str(source.relative_to(root)), "destination": str(target.relative_to(root)), "sha256": sha(source)})
            for row in (json.loads(s) for s in source.read_text(encoding="utf-8").splitlines() if s.strip()):
                role, key = ("solver", "calls") if suffix == ".jsonl" else ("judge", "judge_calls")
                cache_keys[role].update(c["key"] for c in row[key])
    # Judge cache may be reused only if a new answer yields an identical request.
    cache_keys["judge"].update(p.stem for p in (parent_dir / "cache/judge").glob("*.json"))
    for role, keys in cache_keys.items():
        for key in keys:
            checked_copy(parent_dir / "cache" / role / (key + ".json"), output / "cache" / role / (key + ".json"))
    for name in ("direct_context.txt", "direct_context.json"):
        checked_copy(parent_dir / name, output / name)
    for source in (parent_dir / "indexes").glob("*"):
        if source.is_file():
            checked_copy(source, output / "indexes" / source.name)
            reused.append({"source": str(source.relative_to(root)), "destination": str((output / "indexes" / source.name).relative_to(root)),
                           "sha256": sha(source), "kind": "question-independent unchanged index"})
    if not lineage_path.exists():
        write(lineage_path, {"frozen_at": datetime.now().astimezone().isoformat(), "identity": identity,
                             "reused_files": reused, "copied_request_cache_keys": {k: sorted(v) for k, v in cache_keys.items()},
                             "runner_sha256": sha(Path(__file__)), "repar_predictions": "fresh condition, no parent repar predictions/grades copied"})
    return config, output, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/math_2025_national_i_budget.json")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if config_path.is_absolute() or ".." in config_path.parts:
        raise ValueError("Config must be a repository-relative path")
    config, output, runtime = prepare(root, config_path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(runtime / "src"), str(runtime)])

    def phase(name, *extra):
        command = [sys.executable, "-X", "utf8", str(runtime / "scripts/evaluate_math.py"),
                   "--config", config_path.as_posix(), "--phase", name, *extra]
        started = datetime.now().astimezone().isoformat()
        result = subprocess.run(command, cwd=runtime, env=environment)
        with (output / "budget_runs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"started_at": started, "phase": name, "exit_code": result.returncode,
                                     "runtime": str(runtime), "runner_sha256": sha(Path(__file__))}) + "\n")
        if result.returncode:
            raise SystemExit(result.returncode)

    phase("prepare")
    if args.prepare_only:
        return
    phase("generate", "--method", "repar")
    files = {m + ".jsonl": sha(output / (m + ".jsonl")) for m in config["methods"]}
    manifest_path = output / "generation_manifest.json"
    manifest = {"prediction_sha256": files, "counts": {m + ".jsonl": 19 for m in config["methods"]}, "total": 57,
                "note": "19 new RePAR predictions; direct/BM25 historical predictions copied unchanged. Locked before judging."}
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("Locked derived predictions changed")
    if not manifest_path.exists():
        write(manifest_path, manifest)
    phase("grade")
    phase("report")


if __name__ == "__main__":
    main()
