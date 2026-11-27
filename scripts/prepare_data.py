"""Verify the exact ModelScope snapshot; create a label-free manifest."""
import hashlib
import json
from collections import Counter
from pathlib import Path

path = Path("data/longbench_v2.json")
part = path.with_suffix(".json.part")
expected = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
source = path if path.exists() else part
actual = hashlib.sha256(source.read_bytes()).hexdigest()
assert actual == expected, (actual, expected)
if source == part:
    part.replace(path)
data = json.loads(path.read_text(encoding="utf-8"))
subset = [x for x in data if x["domain"].lower().replace("-", " ").startswith("multi")]
manifest = {
    "source": "https://www.modelscope.cn/datasets/ZhipuAI/LongBench-v2",
    "revision": "6a226395b12a1438f40dfabbe0c5ea55f6a102bc", "sha256": actual,
    "total": len(data), "domains": dict(Counter(x["domain"] for x in data)),
    "multi_doc_count": len(subset), "lengths": dict(Counter(x["length"] for x in subset)),
    "sub_domains": dict(Counter(x["sub_domain"] for x in subset)),
    "samples": [{k: x[k] for k in ("_id", "domain", "sub_domain", "difficulty", "length")}
                | {"context_chars": len(x["context"]), "context_sha256": hashlib.sha256(
                    x["context"].encode()).hexdigest()} for x in subset],
}
Path("artifacts/dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in manifest.items() if k != "samples"}, indent=2))
