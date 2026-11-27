"""Download the exact dataset, GGUFs and Windows CUDA runtime used in this run."""
import argparse
import hashlib
import zipfile
from pathlib import Path

import requests

from repar.io import read_json


def download(url, path, expected):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == expected:
                print("Verified", path, flush=True)
                return
        raise RuntimeError(f"Existing file checksum mismatch: {path}")
    part = path.with_suffix(path.suffix+".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        # Servers may ignore Range. Start over if they return the full object.
        append = offset and r.status_code == 206
        with part.open("ab" if append else "wb") as f:
            for chunk in r.iter_content(4*1024*1024):
                f.write(chunk)
    with part.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"Downloaded checksum mismatch {path}: {actual}")
    part.replace(path)
    print("Downloaded and verified", path, flush=True)


p = argparse.ArgumentParser()
p.add_argument("--dataset-only", action="store_true")
args = p.parse_args()
download("https://www.modelscope.cn/api/v1/datasets/ZhipuAI/LongBench-v2/repo?Revision=6a226395b12a1438f40dfabbe0c5ea55f6a102bc&FilePath=data.json",
         "data/longbench_v2.json", "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2")
if args.dataset_only:
    raise SystemExit(0)
for manifest, model in (("model_manifest", "Qwen3.5-0.8B"), ("mtp_model_manifest", "Qwen3.5-0.8B-MTP")):
    m = read_json(f"artifacts/deployment/{manifest}.json")
    download(f"{m['source']}/resolve/{m['revision']}/{m['filename']}", f"models/{model}/{m['filename']}", m["sha256"])
assets = [
    ("llama-b12500-bin-win-cuda-12.4-x64.zip", "1f98033fd54c5259bd0a2e8b06fb13c38e5a2774bd203c043c902615e737db7b"),
    ("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
]
server = Path("tools/llama.cpp/b12500/llama-server.exe")
for name, sha in assets:
    archive = Path("tools/llama.cpp")/name
    download("https://github.com/ggml-org/llama.cpp/releases/download/b12500/"+name, archive, sha)
    if not server.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(server.parent)
    elif name.startswith("cudart"):
        # Do not overwrite a loaded DLL during an experiment.
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (server.parent/member.filename).exists():
                    z.extract(member, server.parent)
