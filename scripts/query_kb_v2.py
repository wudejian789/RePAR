"""Generic CLI for the source tree and entity graph, without benchmark fields."""
import argparse,json
from pathlib import Path
from repar.index_v2 import KnowledgeIndex
from repar.io import read_json,write_json

p=argparse.ArgumentParser()
p.add_argument("--context",required=True)
p.add_argument("--tool",required=True,choices=["tree_search","tree_browse","read","graph_entities","graph_neighbors"])
p.add_argument("--params",default="{}")
p.add_argument("--config",default="configs/repar_v2.json")
p.add_argument("--export")
a=p.parse_args()
kb=KnowledgeIndex.cached(Path(a.context).read_text(encoding="utf-8"),read_json(a.config))
print(json.dumps(kb.execute(a.tool,json.loads(a.params)),ensure_ascii=False,indent=2))
if a.export:
    g=kb.entity_graph
    write_json(a.export,{"stats":kb.stats,"tree":kb.nodes,"spans":kb.spans,"entities":g.entities,
                         "propositions":g.propositions,"relations":g.relations,"passage_entities":g.passage_entities})
