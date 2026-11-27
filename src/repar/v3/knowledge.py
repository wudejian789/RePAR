"""Query contracts on the existing, unchanged tree/entity/source indexes."""
from __future__ import annotations

from .contracts import search_query


class KnowledgeAPI:
    def __init__(self, index):
        self.index = index
        self.context = index.context
        self.context_hash = index.context_hash
        self.stats = index.stats

    def execute(self, tool, parameters, task):
        p = dict(parameters)
        if tool in {"tree_search", "graph_entities", "graph_neighbors"}:
            p["query"] = search_query(p.get("query", ""), task)
        observation = self.index.execute(tool, p)
        records = observation.get("results", [])
        # BM25 zero means no lexical match. Do not apply this to graph path scores.
        if tool == "tree_search":
            observation["results"] = [r for r in records if r.get("score", 0) > 0]
        if tool == "graph_neighbors":
            excluded = set(p.get("exclude_passages", []))
            observation["results"] = [r for r in records if r["id"] not in excluded]
            observation["propositions"] = [r for r in observation.get("propositions", []) if r["passage_id"] not in excluded]
            observation["passage_ids"] = [r["id"] for r in observation["results"]]
        observation["status"] = "ok" if observation.get("results") or observation.get("entities") or observation.get("children") else "no_match"
        observation["executed_parameters"] = p
        observation["capabilities"] = {"text_search": "lexical; immutable original-task anchors combined with current focus",
                                       "graph_extractor": self.index.stats.get("extractor", {}),
                                       "relations": "automatically extracted; co-mention is navigation, not entailment"}
        return observation
