"""Ordered source tree and a separate, many-to-many entity knowledge graph."""
from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path

from .entity_graph import EntityGraph
from .index import KnowledgeIndex as PassageIndex
from .io import digest, write_json


class KnowledgeIndex(PassageIndex):
    @classmethod
    def cached(cls, context, config, directory="artifacts/indexes_v2"):
        key = digest(context+json.dumps({k:config.get(k) for k in
                     ("chunk_words","tree_fanout","graph_model","index_version")},sort_keys=True))
        path = Path(directory)/(key+".pkl")
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        # Reuse only the ordered partition/tree/BM25. The old passage graph is
        # discarded and does not participate in any v2 graph query.
        legacy = {**config,"index_version":"ordered-extractive-v1"}
        base = PassageIndex.cached(context,legacy)
        index = cls.__new__(cls)
        index.__dict__.update(base.__dict__)
        index.config = config
        index._build_graph()
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as f:
            pickle.dump(index,f,pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
        write_json(path.with_suffix(".stats.json"),index.stats)
        return index

    def __init__(self,context,config):
        super().__init__(context,config)
        self._build_graph()

    def _build_graph(self):
        started = time.perf_counter()
        self.entity_graph = EntityGraph(self.texts,self.spans,self.config.get("graph_model","en_core_web_sm"),
                                       progress=lambda n,total: print(f"GRAPH {self.context_hash[:10]} {n}/{total} leaves",flush=True))
        self.edges = [[] for _ in self.texts]  # no legacy passage-similarity graph
        graph = self.entity_graph
        self.stats = {**self.stats,"version":self.config["index_version"],
                      "entities":len(graph.entities),"propositions":len(graph.propositions),
                      "relations":len(graph.relations),"knowledge_nodes":len(graph.entities),
                      "edges":len(graph.relations),"graph_build_seconds":time.perf_counter()-started,
                      "tree_build_seconds":self.stats["build_seconds"],"extractor":graph.extractor}
        self.stats["build_seconds"] = self.stats["tree_build_seconds"]+self.stats["graph_build_seconds"]

    def record(self,i,score=0,preview=False):
        record = super().record(i,score,preview)
        if hasattr(self,"entity_graph"):
            ids = self.entity_graph.passage_entities[i]
            record.update({"entity_ids":ids,"entities":[self.entity_graph.describe(e) for e in ids[:24]],
                           "proposition_ids":self.entity_graph.passage_propositions[i],
                           "source_bundle_id":record.pop("knowledge_id")})
        return record

    def graph_entities(self,query,top_k=8):
        entities = self.entity_graph.search(query,max(1,min(24,int(top_k))))
        return {"operator":"graph_entities","entities":entities,"results":[]}

    def graph_neighbors(self,seeds,query="",hops=1,top_k=8,kinds=None,direction="both",exclude_passages=None,
                        *, eligible_edges_first=False):
        if not seeds:
            raise ValueError("Select at least one existing E ID returned by graph_entities or source passages")
        if direction not in {"both","incoming","outgoing"}:
            raise ValueError("direction must be both, incoming or outgoing")
        limits = {"eligible_edges_first": True} if eligible_edges_first else {}
        graph = self.entity_graph.neighbors(seeds,query,hops,max(1,min(24,int(top_k))),kinds,direction,exclude_passages,**limits)
        return {"operator":"graph_neighbors","seeds":seeds,**graph,
                "results":[self.record(self._leaf_id(cid),graph["scores"].get(cid,0)) for cid in graph["passage_ids"]]}

    def graph_search(self,query="",seeds=None,hops=1,top_k=8,relations=None):
        entity_ids = []
        for seed in seeds or []:
            if str(seed).startswith("E"):
                entity_ids.append(self.entity_graph.entity(seed)["id"])
            elif re.fullmatch(r"C\d+",str(seed)):
                entity_ids.extend(self.entity_graph.passage_entities[self._leaf_id(seed)][:4])
            else:
                raise ValueError("Graph seeds must be existing E entity IDs or C source passage IDs")
        if not entity_ids:
            entity_ids = [e["id"] for e in self.entity_graph.search(query,4)]
        if not entity_ids:
            return {"operator":"graph_search","entities":[],"relations":[],"propositions":[],"paths":[],"results":[]}
        return self.graph_neighbors(list(dict.fromkeys(entity_ids))[:8],query,hops,top_k)

    def execute(self,name,parameters,allow_graph=True):
        functions = {"tree_search":self.tree_search,"tree_browse":self.tree_browse,"read":self.read}
        if allow_graph:
            functions.update({"graph_entities":self.graph_entities,"graph_neighbors":self.graph_neighbors,"graph_search":self.graph_search})
        if name not in functions:
            raise ValueError(f"Unknown or disabled operator: {name}")
        return functions[name](**parameters)
