"""Audit every index and time real queries without reading benchmark answers."""
import time

import numpy as np

from repar.index import KnowledgeIndex
from repar.io import load_public_samples, public_item, read_json, write_json

config = read_json("configs/experiment.json")
results = []
for sample in load_public_samples("data/longbench_v2.json"):
    item = public_item(sample)
    index = KnowledgeIndex.cached(item["context"], config)
    assert index.spans[0][0] == 0 and index.spans[-1][1] == len(item["context"])
    assert all(a[1] == b[0] for a, b in zip(index.spans, index.spans[1:]))
    assert len(index.spans) == len(index.knowledge) == len(index.edges)
    for n in index.nodes:
        children = [index.nodes[c] for c in n["children"]]
        if children:
            assert children[0]["lo"] == n["lo"] and children[-1]["hi"] == n["hi"]
            assert all(a["hi"] == b["lo"] for a, b in zip(children, children[1:]))
        else:
            assert n["hi"] == n["lo"] + 1
    claims = 0
    for k in index.knowledge:
        for claim in k["claims"]:
            assert index.context[claim["start"]:claim["end"]].strip() == claim["text"]
            claims += 1
    for i, edges in enumerate(index.edges):
        for edge in edges:
            assert 0 <= edge["target"] < len(index.spans)
            assert edge["provenance"] == [f"C{i}", f"C{edge['target']}"]
    t = time.perf_counter()
    found = index.tree_search(item["question"], top_k=8)
    tree_ms = (time.perf_counter() - t) * 1000
    seeds = [r["id"] for r in found["results"][:3]]
    t = time.perf_counter()
    related = index.graph_search(item["question"], seeds=seeds, hops=2, top_k=8)
    graph_ms = (time.perf_counter() - t) * 1000
    results.append({"_id": item["_id"], "context_sha256": index.context_hash,
                    "leaves": len(index.spans), "claims_audited": claims,
                    "tree_query_ms": tree_ms, "graph_query_ms": graph_ms,
                    "tree_nodes_visited": found["visited_tree_nodes"],
                    "tree_results": [r["id"] for r in found["results"]],
                    "graph_results": [r["id"] for r in related["results"]]})

summary = {"samples": len(results), "all_invariants_passed": True,
           "claims_audited": sum(r["claims_audited"] for r in results),
           "tree_query_mean_ms": float(np.mean([r["tree_query_ms"] for r in results])),
           "tree_query_p95_ms": float(np.percentile([r["tree_query_ms"] for r in results], 95)),
           "graph_query_mean_ms": float(np.mean([r["graph_query_ms"] for r in results])),
           "graph_query_p95_ms": float(np.percentile([r["graph_query_ms"] for r in results], 95)),
           "tree_nodes_visited_mean": float(np.mean([r["tree_nodes_visited"] for r in results])),
           "timing_protocol": "One question-only top-8 tree query and one top-8 two-hop graph query per sample; CPU index already loaded; no quality claim or cold-start estimate."}
write_json("artifacts/structure_audit.json", {"summary": summary, "samples": results})
print(summary)
