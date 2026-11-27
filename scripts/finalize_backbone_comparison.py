"""Wait for frozen 27B conditions, audit each, then build the paired report."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/repar_27b_comparison_20261119"


def run(script,target,log):
    with log.open("w",encoding="utf-8") as stream:
        subprocess.run([sys.executable,"-X","utf8",script,*target],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    for method,script in (("v2","artifacts/maintenance/audit_backbone_v2.py"),("v3","artifacts/maintenance/audit_v3_core.py")):
        path=OUT/method
        while not (path/"summary.json").exists():time.sleep(10)
        run("scripts/run_frozen_jsonl_compat.py",["--experiment",str(path.relative_to(ROOT)),
            "--script",script,"--","--experiment",str(path.relative_to(ROOT))],path/"audit.log")
        print("AUDITED",method,flush=True)
    while not (OUT/"generation_complete.json").exists():time.sleep(2)
    run("artifacts/maintenance/report_backbone_comparison.py",[],OUT/"report.log")
    print("Full paired report and completion written",flush=True)


if __name__=="__main__":main()
