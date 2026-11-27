"""Generic local tree/graph query API, independent of LongBench."""
import argparse
import json
from pathlib import Path

from repar.index import KnowledgeIndex
from repar.io import read_json, write_json

p = argparse.ArgumentParser()
p.add_argument("--context", required=True, help="UTF-8 knowledge text")
p.add_argument("--tool", choices=["tree_browse", "tree_search", "graph_search", "read"], required=True)
p.add_argument("--params", default="{}", help="JSON parameter object")
p.add_argument("--config", default="configs/experiment.json")
p.add_argument("--export", help="Optional JSON export of tree, propositions, provenance and graph")
args = p.parse_args()
index = KnowledgeIndex.cached(Path(args.context).read_text(encoding="utf-8"), read_json(args.config))
result = index.execute(args.tool, json.loads(args.params))
print(json.dumps(result, ensure_ascii=False, indent=2))
if args.export:
    write_json(args.export, {"stats": index.stats, "tree": index.nodes, "spans": index.spans,
                             "knowledge": index.knowledge, "edges": index.edges})
