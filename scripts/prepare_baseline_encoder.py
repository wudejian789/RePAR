"""Download a pinned small, frozen CPU encoder, recording exact file hashes."""
from pathlib import Path
import hashlib
import json

import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    target = ROOT / "models/all-MiniLM-L6-v2"
    target.mkdir(parents=True, exist_ok=True)
    lock = target / "download_manifest.json"
    previous = json.loads(lock.read_text()) if lock.exists() else None
    revision = previous["revision"] if previous else requests.get(
        f"https://huggingface.co/api/models/{MODEL}", timeout=30).json()["sha"]
    paths = ["config.json", "modules.json", "sentence_bert_config.json",
             "config_sentence_transformers.json", "1_Pooling/config.json",
             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "vocab.txt", "model.safetensors", "README.md"]
    files = []
    for name in paths:
        path = target / name
        url = f"https://huggingface.co/{MODEL}/resolve/{revision}/{name}"
        if not path.exists():
            response = requests.get(url, timeout=(15, 180))
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        files.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "bytes": path.stat().st_size})
        print(name, path.stat().st_size, flush=True)
    lock.write_text(json.dumps({"model": MODEL, "revision": revision, "files": files,
        "usage": "Frozen CPU retriever encoder; no training; Qwen generator unchanged"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
