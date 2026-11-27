"""Audit both graph layers, exact source provenance and graph traversal on all125."""
import time
from collections import Counter

from repar.index_v2 import KnowledgeIndex
from repar.io import load_public_samples,read_json,write_json


def audit():
    config=read_json("configs/repar_v2.json")
    rows=[]
    for number,item in enumerate(load_public_samples("data/longbench_v2.json"),1):
        kb=KnowledgeIndex.cached(item["context"],config)
        g=kb.entity_graph
        assert len(g.passage_entities)==len(kb.spans)
        assert kb.spans[0][0]==0 and kb.spans[-1][1]==len(kb.context)
        assert all(a[1]==b[0] for a,b in zip(kb.spans,kb.spans[1:]))
        mentions=0
        for entity in g.entities:
            assert entity["passage_ids"]
            for m in entity["mentions"]:
                assert kb.context[m["start"]:m["end"]]==m["text"]
                leaf=kb._leaf_id(m["passage_id"])
                assert entity["id"] in g.passage_entities[leaf]
                assert entity["id"] in g.propositions[int(m["proposition_id"][1:])]["entity_ids"]
                mentions+=1
        for proposition in g.propositions:
            assert kb.context[proposition["start"]:proposition["end"]]==proposition["text"]
            leaf=kb._leaf_id(proposition["passage_id"])
            assert kb.spans[leaf][0]<=proposition["start"]<proposition["end"]<=kb.spans[leaf][1]
        for relation in g.relations:
            assert relation["subject"]!=relation["object"]
            for pid in relation["proposition_ids"]:
                p=g.propositions[int(pid[1:])]
                assert relation["id"] in p["relation_ids"]
                assert {relation["subject"],relation["object"]}<=set(p["entity_ids"])
                assert p["passage_id"] in relation["passage_ids"]
        # This operational check does not read the QA or claim retrieval quality.
        candidates=[e for e in g.entities if len(e["passage_ids"])>1 and g.adjacency[e["id"]]]
        cross=0;ms=0
        if candidates:
            e=candidates[0]; t=time.perf_counter()
            o=kb.graph_neighbors([e["id"]],hops=2,top_k=8)
            ms=(time.perf_counter()-t)*1000
            cross=len({r["id"] for r in o["results"]})
        row={"_id":item["_id"],**kb.stats,"mentions_audited":mentions,
             "cross_passage_entities":sum(len(e["passage_ids"])>1 for e in g.entities),
             "relation_kinds":dict(Counter(r["kind"] for r in g.relations)),
             "graph_neighbor_ms":ms,"graph_neighbor_source_count":cross}
        rows.append(row)
        print("AUDIT",number,"/125",item["_id"],"entities",len(g.entities),"relations",len(g.relations),flush=True)
    summary={"samples":len(rows),"all_invariants_passed":True,
             **{k:sum(r[k] for r in rows) for k in ("entities","relations","propositions","mentions_audited","cross_passage_entities","graph_build_seconds")}}
    write_json("artifacts/structure_audit_v2.json",{"summary":summary,"samples":rows})
    print(summary,flush=True)
    return summary


if __name__=="__main__":audit()
