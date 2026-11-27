from repar.v4.knowledge import KnowledgeAPI
from repar.v4.source_quality import unavailable_leaves
from test_v4 import Index


def test_unavailable_passage_is_replaced_by_a_usable_lexical_candidate():
    class RankedIndex(Index):
        def __init__(self):
            super().__init__()
            self.spans = [(self.record(i)["start"], self.record(i)["end"]) for i in range(2)]

        def bm25(self, query, top_k, scope):
            return [(0, 4), (1, 2)][:top_k]

        def record(self, i, score=None):
            row = super().record(i)
            return row if score is None else {**row, "score": score}

        def execute(self, tool, parameters):
            if tool == "tree_search":
                return {"operator": tool, "results": [self.record(0)]}
            return super().execute(tool, parameters)

    index = RankedIndex()
    start, end = index.spans[0]
    excluded = unavailable_leaves(index, [(start, start + 20), (start + 10, end)])
    assert excluded == {"C0"}
    api = KnowledgeAPI(index, r3_retrieval=True, unavailable_sources=excluded)
    results = api.execute("tree_search", {"query": "Willow", "mode": "hybrid", "top_k": 1})
    assert [row["id"] for row in results["results"]] == ["C1"]
    hidden = api.execute("read", {"ids": ["C0"], "radius": 0})
    assert hidden["status"] == "no_match" and hidden["results"] == []

    # Explicitly advancing past a seen record must also refill the result set.
    index._leaf_id = lambda cid: int(cid[1:])
    api = KnowledgeAPI(index, r3_retrieval=True)
    results = api.execute("tree_search", {"query": "Willow", "mode": "hybrid", "top_k": 1, "exclude_passages": ["C0"]})
    assert [row["id"] for row in results["results"]] == ["C1"]
