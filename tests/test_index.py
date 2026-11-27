from repar.index import KnowledgeIndex, split_spans
from repar.io import public_item


CONFIG = {"chunk_words": 35, "tree_fanout": 4, "graph_neighbors": 3, "index_version": "test"}


def test_lossless_order_and_provenance():
    text = ("Alpha study reports that treatment improves patient outcomes. " * 40 + "\n\n" +
            "Beta study however observes a reduction in access because budgets declined in 2022. " * 35)
    spans = split_spans(text, 35)
    assert "".join(text[a:b] for a, b in spans) == text
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    index = KnowledgeIndex(text, CONFIG)
    for node in index.nodes:
        children = [index.nodes[c] for c in node["children"]]
        if children:
            assert children[0]["lo"] == node["lo"]
            assert children[-1]["hi"] == node["hi"]
            assert all(a["hi"] == b["lo"] for a, b in zip(children, children[1:]))
    for k in index.knowledge:
        for claim in k["claims"]:
            assert text[claim["start"]:claim["end"]].strip() == claim["text"]
    assert any("causal_cue" in k["logic_cues"] for k in index.knowledge)


def test_cross_passage_bridge_and_scope():
    context = "\n\n".join([
        "The Delta Commission developed the Silver Protocol to monitor river pollution and report standards. " * 3,
        "The museum catalog lists paintings with extensive annotations about historic architecture and famous artists. " * 3,
        "Silver Protocol was adopted by Harbor Council in 2024 because freshwater safety required consistent monitoring. " * 3,
        "The electronics handbook describes unrelated transistor circuits and computational devices with electrical measurements. " * 3,
    ])
    index = KnowledgeIndex(context, CONFIG)
    first = index.tree_search("Delta Commission Silver Protocol", top_k=1)["results"][0]
    assert "Delta Commission" in first["text"]
    graph = index.graph_search("Harbor Council", seeds=[first["id"]], hops=2, top_k=12)
    assert any("Harbor Council" in r["text"] for r in graph["results"])
    assert all(r["edge"]["provenance"] for r in graph["results"])
    subtree = next(n for n in index.nodes if n["children"] and n["id"] != "T0")
    scoped = index.tree_search("Silver Protocol", scope=subtree["id"], top_k=24)
    assert all(subtree["lo"] <= int(r["id"][1:]) < subtree["hi"] for r in scoped["results"])


def test_invalid_operators_and_label_boundary():
    import pytest
    index = KnowledgeIndex("A meaningful document about bridge construction and road safety. " * 8, CONFIG)
    with pytest.raises(ValueError):
        index.execute("graph_search", {"query": "bridge"}, allow_graph=False)
    with pytest.raises(ValueError):
        index.read(["C999999"])
    assert "answer" not in public_item({"answer": "D", "question": "Which?", "context": "source"})
