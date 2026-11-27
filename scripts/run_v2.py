"""Run full v2 experiments as the independent, context-only graph build proceeds."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from repar.io import read_json,write_json


def main():
    output=Path("artifacts/experiment_v2")
    output.mkdir(exist_ok=True)
    paths=[*Path("src").rglob("*.py"),Path("configs/repar_v2.json"),*Path("tests").glob("test_v2.py")]
    manifest=Path("artifacts/implementation_manifest_v2.json")
    if not manifest.exists():
        write_json(manifest,{"time":time.strftime('%FT%T%z'),
                            "files":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                            "protocol":"plan -> parameterized action -> real observation -> reflection",
                            "selection":"all125; standard Q4; no per-case rules or label-based parameter selection",
                            "prior_exposure":"v1 scores are known; v2 protocol checks used synthetic general tasks"})
    for method in ("repar","repar_tree"):
        command=[sys.executable,"-X","utf8","-m","repar.cli","run","--method",method,
                 "--config","configs/repar_v2.json","--output",str(output),"--wait-for-indexes"]
        with (output/(method+".log")).open("a",encoding="utf-8") as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,"-X","utf8","scripts/audit_v2.py"],check=True)
    subprocess.run([sys.executable,"-X","utf8","scripts/report_v2.py"],check=True)
    subprocess.run([sys.executable,"-X","utf8","-m","webserver.build_data"],check=True)
    write_json("artifacts/v2_complete.json",{"time":time.strftime('%FT%T%z'),"v2_predictions":250,
                                            "reused_baselines":"standard Q4 direct/BM25 with unchanged source leaf partition"})


if __name__=="__main__":main()
