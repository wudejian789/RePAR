"""CPU-only, answer-independent indexing, concurrent with remote model inference."""
from repar.index import KnowledgeIndex
from repar.io import load_public_samples, read_json

config = read_json("configs/experiment.json")
samples = load_public_samples("data/longbench_v2.json")
for i, item in enumerate(samples):
    index = KnowledgeIndex.cached(item["context"], config)
    print(i + 1, len(samples), item["_id"], index.stats, flush=True)
