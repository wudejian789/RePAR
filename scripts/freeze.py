"""Record code and prompt checksums before looking at aggregate scores."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

paths = [*Path("src").rglob("*.py"), *Path("configs").glob("*.json"),
         *Path("third_party/prompts").glob("*.txt"), *Path("tests").glob("*.py")]
manifest = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
            "packages": subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"], text=True).splitlines(),
            "selection": "All 125 Multi-Document QA samples, no tuning on benchmark labels",
            "conditions": ["direct", "bm25", "repar", "repar_tree"]}
Path("artifacts/implementation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Implementation frozen:", len(paths), "files")
