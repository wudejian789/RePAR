"""Check whether MTP and standard quants really share identical target tensors."""
import hashlib
from collections import Counter
from pathlib import Path

from gguf import GGUFReader

from repar.io import write_json

paths = {
    "standard": "models/Qwen3.5-0.8B/Qwen3.5-0.8B-Q4_K_M.gguf",
    "mtp": "models/Qwen3.5-0.8B-MTP/Qwen3.5-0.8B-Q4_K_M.gguf",
}
models = {}
for label, path in paths.items():
    reader = GGUFReader(path)
    tensors = {}
    for tensor in reader.tensors:
        tensors[tensor.name] = {"shape": tensor.shape.tolist(), "type": str(tensor.tensor_type),
                                "bytes": int(tensor.n_bytes), "sha256": hashlib.sha256(tensor.data).hexdigest()}
    models[label] = tensors
common = sorted(set(models["standard"]) & set(models["mtp"]))
different = [n for n in common if models["standard"][n] != models["mtp"][n]]
summary = {"common_tensors": len(common), "identical_tensors": len(common)-len(different),
           "different_target_tensors": different,
           "standard_only": sorted(set(models["standard"]) - set(common)),
           "mtp_only": sorted(set(models["mtp"]) - set(common)),
           "all_common_target_tensors_identical": not different}
write_json("artifacts/deployment/gguf_tensor_comparison.json", {"summary": summary, "models": models})
print(summary)
