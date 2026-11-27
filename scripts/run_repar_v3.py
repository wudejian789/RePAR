"""Freeze v3 before scoring; always execute the frozen snapshot in isolation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_experiment_core.json")
    parser.add_argument("--phase", choices=("math", "longbench", "report", "all"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/args.config).read_text(encoding="utf-8"))
    protocol_path = root/config["output"]/"protocol.json"
    if not protocol_path.exists():
        from repar.benchmarks.v3_suite import freeze
        freeze(args.config)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    snapshot = protocol_path.parent/"source"
    runtime = Path(tempfile.mkdtemp(prefix="repar_v3_"))
    import hashlib
    for name, expected in protocol["identity"]["files"].items():
        source = snapshot/name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected: raise ValueError(f"Snapshot changed: {name}")
        target = runtime/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    for name in (("data", "artifacts", "models") if protocol['identity']['solver'].get('source_similarity') else ("data", "artifacts")):
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(runtime/name), str(root/name)], check=True, stdout=subprocess.DEVNULL)
        else: (runtime/name).symlink_to(root/name, target_is_directory=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(runtime/"src"), str(runtime)])
    command = [sys.executable, "-X", "utf8", "-c", "from repar.benchmarks.v3_suite import main; main()",
               "--config", args.config, "--phase", args.phase]
    result = subprocess.run(command, cwd=runtime, env=environment)
    raise SystemExit(result.returncode)


if __name__ == "__main__": main()
