"""Run unchanged frozen evidence with a narrow JSONL delimiter compatibility fix."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--experiment")
    parser.add_argument("--phase",default="all")
    parser.add_argument("--script")
    parser.add_argument("script_args",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/args.config).read_text(encoding="utf-8")) if args.config else None
    output=root/(config["output"] if config else args.experiment)
    protocol=json.loads((output/"protocol.json").read_text(encoding="utf-8"))
    if args.script:
        sys.path[:0]=[str(output/"source/src"),str(output/"source")]
    else:
        runtime=Path(tempfile.mkdtemp(prefix="repar_jsonl_compat_"))
        for name,expected in protocol["identity"]["files"].items():
            source=output/"source"/name
            if sha(source)!=expected:raise ValueError("Frozen source changed: "+name)
            target=runtime/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        for name in ("data","artifacts"):
            if os.name=="nt":subprocess.run(["cmd","/c","mklink","/J",str(runtime/name),str(root/name)],check=True,stdout=subprocess.DEVNULL)
            else:(runtime/name).symlink_to(root/name,target_is_directory=True)
        os.chdir(runtime);sys.path[:0]=[str(runtime/"src"),str(runtime)]
    import repar.io
    def read_jsonl(path):
        path=Path(path)
        if not path.exists():return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    repar.io.read_jsonl=read_jsonl
    record={"time":datetime.now().astimezone().isoformat(),"launcher_sha256":sha(Path(__file__)),
        "override":"repar.io.read_jsonl: split only physical newline, preserve literal U+2028/U+2029 within JSON strings",
        "frozen_source_unchanged":True,"prediction_files_rewritten":False,"config":args.config,"script":args.script,"phase":args.phase}
    with (output/"jsonl_reader_compatibility.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(record,ensure_ascii=False)+"\n")
    if args.script:
        sys.argv=[args.script,*[arg for arg in args.script_args if arg!="--"]]
        runpy.run_path(str(root/args.script),run_name="__main__")
    else:
        sys.argv=["frozen_suite","--config",args.config,"--phase",args.phase]
        from repar.benchmarks.v3_suite import main as suite_main
        suite_main()


if __name__=="__main__":main()
