import json

from repar.agent import answer
from repar.evaluation import extract_answer
from repar.index import KnowledgeIndex


class FakeClient:
    def __init__(self):
        self.step = 0
        self.requests = []

    def complete(self, messages, max_tokens, json_mode=False, tag=""):
        self.requests.append(messages)
        if not json_mode:
            return "The correct answer is (B)"
        self.step += 1
        if self.step == 1:
            return json.dumps({"reflection": "No evidence yet", "plan": "Locate the first period",
                               "policy_update": {"focus": "first period"}, "finish": False,
                               "tool_calls": [{"name": "tree_search", "parameters": {"query": "river", "top_k": 1}}]})
        if self.step == 2:
            state = json.loads(messages[1]["content"])
            assert "C0" in state["current_evidence"]
            return json.dumps({"reflection": "Need the related policy source", "plan": "Follow an evidence bridge",
                               "policy_update": {"focus": "second period"}, "finish": False,
                               "tool_calls": [{"name": "graph_search", "parameters": {"seeds": ["K0"], "query": "safety"}}]})
        return json.dumps({"reflection": "Evidence now sufficient", "finish": True, "tool_calls": []})


def test_dynamic_execution_reflection_and_termination(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompts = tmp_path / "third_party/prompts"
    prompts.mkdir(parents=True)
    (prompts / "0shot_rag.txt").write_text("$DOC$\n$Q$\n$C_A$\n$C_B$\n$C_C$\n$C_D$", encoding="utf-8")
    config = {"chunk_words": 25, "tree_fanout": 4, "graph_neighbors": 3,
              "index_version": "test", "evidence_word_budget": 1000,
              "react_max_steps": 5, "react_max_tokens": 200, "answer_max_tokens": 128}
    index = KnowledgeIndex("River monitoring supports safe drinking water because regulations require it. " * 12, config)
    client = FakeClient()
    item = {"_id": "synthetic", "question": "What follows?", **{"choice_" + c: c for c in "ABCD"}}
    result = answer(item, index, client, config)
    assert result["pred"] == "B"
    assert result["rounds"] == 3 and result["tool_calls"] == 2
    assert result["termination"] == "model_sufficient" and not result["fallback"]
    trace = json.loads((tmp_path / result["trace_path"]).read_text(encoding="utf-8"))
    assert trace["policies"][:2] == [{"focus": "first period"}, {"focus": "second period"}]
    assert all("answer" not in m[1]["content"] for m in client.requests[:-1])


def test_official_answer_parser():
    assert extract_answer("**The correct answer is (C)**") == "C"
    assert extract_answer("The correct answer is D") == "D"
    assert extract_answer("Maybe A or B") is None


def test_finish_conflict_and_duplicate_feedback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompts = tmp_path / "third_party/prompts"
    prompts.mkdir(parents=True)
    (prompts / "0shot_rag.txt").write_text("$DOC$\n$Q$", encoding="utf-8")
    config = {"chunk_words": 25, "tree_fanout": 4, "graph_neighbors": 3,
              "index_version": "test", "evidence_word_budget": 1000,
              "react_max_steps": 5, "react_max_tokens": 200, "answer_max_tokens": 128}
    index = KnowledgeIndex("River monitoring supports safety because regulations require it. " * 12, config)
    item = {"_id": "synthetic", "question": "What follows?", **{"choice_" + c: c for c in "ABCD"}}

    class ConflictedClient(FakeClient):
        def complete(self, messages, max_tokens, json_mode=False, tag=""):
            if not json_mode:
                return "The correct answer is (B)"
            self.step += 1
            state = json.loads(messages[1]["content"])
            if self.step <= 2:
                calls = [{"name": "tree_search", "parameters": {"query": "river", "top_k": 1}}]
            elif self.step == 3:
                assert "Repeated successful call" in state["previous_rounds"][-1]["observations"][0]["error"]
                calls = [{"name": "graph_search", "parameters": {"query": "safety", "seeds": ["K0"]}}]
            else:
                assert state["retrieval_rounds"] == 2
                calls = []
            return json.dumps({"tool_calls": calls, "finish": True})

    result = answer(item, index, ConflictedClient(), config)
    trace = json.loads((tmp_path / result["trace_path"]).read_text(encoding="utf-8"))
    assert result["rounds"] == 4 and result["termination"] == "model_sufficient"
    assert trace["trace"][2]["executed"][0]["name"] == "graph_search"


def test_graph_ablation_keeps_planner_and_blocks_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompts = tmp_path / "third_party/prompts"
    prompts.mkdir(parents=True)
    (prompts / "0shot_rag.txt").write_text("$DOC$\n$Q$", encoding="utf-8")
    config = {"chunk_words": 25, "tree_fanout": 4, "graph_neighbors": 3,
              "index_version": "test", "evidence_word_budget": 1000,
              "react_max_steps": 5, "react_max_tokens": 200, "answer_max_tokens": 128}
    index = KnowledgeIndex("River monitoring supports safe drinking water because regulations require it. " * 12, config)
    item = {"_id": "synthetic", "question": "What follows?", **{"choice_" + c: c for c in "ABCD"}}
    a, b = FakeClient(), FakeClient()
    full = answer(item, index, a, config, "repar")
    ablated = answer(item, index, b, config, "repar_tree")
    assert a.requests[0] == b.requests[0]
    t = json.loads((tmp_path / ablated["trace_path"]).read_text(encoding="utf-8"))
    assert "disabled operator" in t["trace"][1]["executed"][0]["error"]
    assert full["pred"] == ablated["pred"] == "B"
