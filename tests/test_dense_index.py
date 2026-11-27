import re

import numpy as np

from repar import dense_index
from repar.benchmarks import v4_suite
from repar.benchmarks.math_exam import task_text
from repar.dense_index import DenseKnowledgeIndex, split_token_spans
from repar.v3.workflow import structured


def test_token_partition_preserves_text_and_caps_both_scripts():
    def tokenizer(text, **options):
        spans = [m.span() for m in re.finditer(r"[A-Za-z]+|[^\s]", text)]
        return {"offset_mapping": spans, "input_ids": list(range(len(spans)))}

    text = "  Willow Museum.\n\n档案馆保存星图。画廊展出瓷器。\n  "
    spans, counts = split_token_spans(text, tokenizer, 5)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(x[1] == y[0] for x, y in zip(spans, spans[1:]))
    assert max(counts) <= 5


def test_dense_tree_and_exact_candidates_obey_observed_scope(monkeypatch):
    index = DenseKnowledgeIndex.__new__(DenseKnowledgeIndex)
    index.context = "ABCD"
    index.texts = list(index.context)
    index.spans = [(i, i+1) for i in range(4)]
    index.knowledge = [{"concepts": [], "logic_cues": []} for _ in index.texts]
    index.matrix = np.array([[0., 1.], [.28, .96], [.8, .6], [1., 0.]], dtype=np.float32)
    index.build_tree(2)
    monkeypatch.setattr(index, "encoder_settings", lambda: ())
    monkeypatch.setattr(dense_index, "query_vector", lambda query: np.array([1., 0.]))
    root = index.tree_search("paraphrase", top_k=2)
    assert [r["id"] for r in root["results"]] == ["C3", "C2"]
    scope = index.nodes[index.leaf_to_tree[1]]["id"]
    scoped = index.tree_search("paraphrase", scope=scope, top_k=2, mode="tree")
    assert [r["id"] for r in scoped["results"]] == ["C1"]
    assert index.tree_search("  ")["results"] == []


def test_r61_hybrid_surfaces_all_exact_hits_when_tree_routing_misses(monkeypatch):
    index = DenseKnowledgeIndex.__new__(DenseKnowledgeIndex)
    index.context = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    index.texts = list(index.context)
    index.spans = [(i, i+1) for i in range(len(index.texts))]
    index.knowledge = [{"concepts": [], "logic_cues": []} for _ in index.texts]
    index.config = {"agent_version": dense_index.R61}
    index.matrix = np.array([[1., i/26] for i in range(26)], dtype=np.float32)
    index.build_tree(4)
    # The root and scoped leaves still exist, but simulate a misleading routing centroid.
    index.tree_matrix[:, 0] = -np.array([node["lo"] for node in index.nodes])
    index.tree_matrix[:, 1] = 0
    monkeypatch.setattr(index, "encoder_settings", lambda: ())
    monkeypatch.setattr(dense_index, "query_vector", lambda query: np.array([1., 1.]))
    result = index.tree_search("relevant clue", top_k=3, beam_width=3)
    expected = ["C25", "C24", "C23"]
    assert set(expected) - set(result["candidate_generation"]["routed_ids"])
    assert result["candidate_generation"]["exact_dense_ids"] == expected
    assert [record["id"] for record in result["results"]] == expected


def test_r61_formats_public_task_and_keeps_a_question_anchor(monkeypatch, tmp_path):
    suite = v4_suite.V4Suite.__new__(v4_suite.V4Suite)
    suite.solver = {"agent_version": dense_index.R61}
    suite.output = tmp_path
    suite.config = {}
    monkeypatch.setattr(v4_suite, "research", lambda task, *_args, **kwargs: (task, kwargs))
    task = task_text({"question_stem": "档案馆如何查阅星图？", "question_type": "single_choice",
                      "choices": [{"tag": "A", "text": "预约"}]})
    actual, payload = suite.research_result(task, None, None, "math", "sample")
    assert actual == task
    assert payload["query_task"] == "档案馆如何查阅星图？\n\nA. 预约"
    assert payload["query_anchor"] == "档案馆如何查阅星图？"


def test_structured_feedback_repairs_a_json_string_with_an_unpaired_surrogate():
    class Client:
        def complete(self, messages, budget, *, tag, **kwargs):
            if tag.endswith("_repair"):
                assert messages[-2]["content"] == r'{"gap":"\ud83c"}'
                return '{"gap":"查找馆藏记录"}'
            return r'{"gap":"\ud83c"}'

    schema = {"type": "object", "properties": {"gap": {"type": "string"}}, "required": ["gap"]}
    assert structured(Client(), [], schema, 128, "reflection") == {"gap": "查找馆藏记录"}
