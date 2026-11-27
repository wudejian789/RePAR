"""Resume all prespecified conditions; fail if any of 125 samples is missing."""
import subprocess
import sys
import argparse
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("methods", nargs="*", default=["direct", "bm25", "repar", "repar_tree"])
p.add_argument("--config", default="configs/experiment.json")
p.add_argument("--output", default="artifacts/experiment")
p.add_argument("--workers", default=1, type=int)
p.add_argument("--skip-report", action="store_true")
args = p.parse_args()
methods = args.methods or ["direct", "bm25", "repar", "repar_tree"]
subprocess.run([sys.executable, "-X", "utf8", "scripts/audit_structure.py"], cwd=root, check=True)
for method in methods:
    stop = root / args.output / (method + ".stop")
    if stop.exists():
        stop.unlink()
    subprocess.run([sys.executable, "-X", "utf8", "-m", "repar.cli", "run", "--method", method,
                    "--workers", str(args.workers), "--config", args.config, "--output", args.output],
                   cwd=root, check=True)
if not args.skip_report:
    subprocess.run([sys.executable, "-X", "utf8", "scripts/report.py", "--root", args.output,
                    "--config", args.config], cwd=root, check=True)
    subprocess.run([sys.executable, "-X", "utf8", "scripts/plot_results.py", "--root", args.output], cwd=root, check=True)
