"""Audit completed27B Direct/BM25 conditions, then report all four methods."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/baselines_27b_20261119"


def main():
    for method in ("direct","bm25"):
        p=OUT/method
        while not (p/"summary.json").exists():time.sleep(10)
        with (p/"audit.log").open("w",encoding="utf-8") as stream:
            subprocess.run([sys.executable,"-X","utf8","artifacts/maintenance/audit_27b_baselines.py",
                "--experiment",str(p.relative_to(ROOT))],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print("AUDITED",method,flush=True)
    while not (OUT/"generation_complete.json").exists():time.sleep(2)
    with (OUT/"report.log").open("w",encoding="utf-8") as stream:
        subprocess.run([sys.executable,"-X","utf8","artifacts/maintenance/report_27b_all_methods.py"],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print("Four-method report completed",flush=True)


if __name__=="__main__":main()
