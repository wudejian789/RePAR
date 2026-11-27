"""Independent retrieval cases and parity checks for the narrowed v3 core."""
import json
from copy import deepcopy

import jsonschema
import pytest

from repar.v3.contracts import request_identity, search_query
from repar.v3.knowledge import KnowledgeAPI
from repar.v3.memory import EvidenceMemory
from repar.v3.workflow import research, decision_schema, ResearchResult


class TokenClient:
    def tokenize(self, text): return list(text)


class Index:
    context = "Harbor Library acquired the Cedar Archive in 2022.\nThe Cedar Archive moved in 2024."
    context_hash = "synthetic"
    stats = {"extractor": {"model": "fixture"}}

    def __init__(self): self.calls = []

    def record(self, number, score=2):
        start = 0 if number == 0 else self.context.index("\n")+1
        end = self.context.index("\n") if number == 0 else len(self.context)
        return {"id": f"C{number}", "start": start, "end": end, "text": self.context[start:end],
                "score": score, "tree_path": ["T0", f"T{number+1}"], "entity_ids": ["E0"],
                "entities": [{"id": "E0", "name": "Cedar Archive"}]}

    def execute(self, tool, params):
        self.calls.append((tool, deepcopy(params)))
        if tool == "tree_browse":
            return {"operator": tool, "node": {"id": 0}, "children": [{"id": 1}, {"id": 2}]}
        if tool == "tree_search":
            return {"operator": tool, "results": [self.record(0, 0 if params["query"] == "quasar" else 2)]}
        if tool == "graph_entities":
            return {"operator": tool, "entities": [{"id": "E0", "name": "Cedar Archive"}], "results": []}
        return {"operator": tool, "results": [self.record(0), self.record(1)]}


def tree_action(**overrides):
    return {"tool": "tree_search", "query": "Cedar Archive acquired", "scope": "T0",
            "top_k": 6, "mode": "hybrid", "beam_width": 9, **overrides}


class Client(TokenClient):
    def __init__(self, actions, stop_after=None):
        self.actions, self.stop_after, self.events = iter(actions), stop_after, []
        self.observations = 0

    def complete(self, messages, budget, **kwargs):
        tag = kwargs["tag"]; body = json.loads(messages[-1]["content"])
        self.events.append((tag, deepcopy(body), deepcopy(kwargs)))
        if "decision" in tag:
            return json.dumps({"objective": "Find source support for the archive task", "action": next(self.actions)})
        self.observations += 1
        assert "observation" in body
        return json.dumps({"assessment": "The actual source observation is available", "gap": "Check source links",
            "lesson": "Use observed references", "supported_sources": body["available_sources"][:2],
            "next": "answer" if self.stop_after == self.observations else "continue"})


def test_identity_keeps_constraints_and_normalizes_only_whitespace():
    identity = lambda q: request_identity("search", {"query": q, "scope": "root"})
    assert identity("x > 0") != identity("x < 0")
    assert identity("relay enabled") != identity("relay not enabled")
    assert identity("before 2020") != identity("after 2020")
    assert identity("A  B") == identity("A B")


def test_query_keeps_original_task_without_rewriting_sources():
    original = "When did Harbor Library acquire the Cedar Archive?"
    assert original in search_query("archive acquisition", original)
    assert search_query(original, original) == original


def test_real_counter_caps_unspaced_source_and_preserves_offsets():
    context = "天文台记录光谱仪的测量结果。" * 200
    memory = EvidenceMemory(context, TokenClient(), 240)
    memory.observe({"results": [{"id": "C0", "start": 0, "end": len(context), "text": context}]}, 0)
    evidence, selected = memory.render("光谱仪")
    assert len(evidence) <= 240 and selected
    assert all(context[r["start"]:r["end"]] == r["text"] for r in selected)
    with pytest.raises(ValueError, match="offsets"):
        memory.observe({"results": [{"id": "C1", "start": 0, "end": 2, "text": "fake"}]}, 1)


def test_zero_match_and_strict_exclude_contracts():
    api = KnowledgeAPI(Index())
    assert api.execute("tree_search", {"query": "quasar"}, "quasar")["status"] == "no_match"
    returned = api.execute("graph_neighbors", {"query": "archive", "exclude_passages": ["C0"]}, "archive")
    assert [r["id"] for r in returned["results"]] == ["C1"]


def test_only_existing_knowledge_tools_and_v2_answer_availability():
    memory, index = EvidenceMemory(Index.context, TokenClient()), Index()
    variants = decision_schema(memory, index)["properties"]["action"]["oneOf"]
    assert {v["properties"]["tool"]["const"] for v in variants} == {"tree_search", "tree_browse", "graph_entities"}
    memory.observe({"results": [index.record(0)]}, 0)
    variants = decision_schema(memory, index)["properties"]["action"]["oneOf"]
    assert {v["properties"]["tool"]["const"] for v in variants} == {
        "tree_search", "tree_browse", "read", "graph_entities", "graph_neighbors", "answer"}


def test_bound_parameters_reject_unknown_ids_and_out_of_budget_values():
    memory, index = EvidenceMemory(Index.context, TokenClient()), Index()
    memory.observe({"results": [index.record(0)]}, 0)
    schema = decision_schema(memory, index)
    for action in (tree_action(scope="T999"), tree_action(top_k=25), tree_action(beam_width=0),
                   {"tool": "read", "refs": ["C999"], "radius": 1}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"objective": "Retrieve sources", "action": action}, schema)


def test_observation_precedes_reflection_and_parameters_are_model_selected(tmp_path):
    index, client = Index(), Client([tree_action()], stop_after=1)
    result = research("When was the Cedar Archive acquired?", index, client,
        {"evidence_token_budget": 500, "controller_evidence_tokens": 200}, event_path=tmp_path/"events.jsonl")
    assert [e[0] for e in client.events] == ["v3_core_decision_1", "v3_core_reflection_1"]
    assert index.calls[0][1]["top_k"] == 6 and index.calls[0][1]["beam_width"] == 9
    assert result.evidence_ids == ["C0"] and result.termination == "model_sufficient"
    assert [json.loads(line)["stage"] for line in (tmp_path/"events.jsonl").read_text().splitlines()] == ["observation", "reflection"]


def test_source_entities_enable_existing_graph_query_without_hidden_route():
    graph = {"tool": "graph_neighbors", "query": "archive move", "refs": ["E0"],
             "top_k": 3, "hops": 1, "direction": "incoming", "kinds": ["predicate"]}
    index, client = Index(), Client([tree_action(), graph], stop_after=2)
    result = research("Trace the Cedar Archive acquisition and move.", index, client, {})
    assert [name for name, _ in index.calls] == ["tree_search", "graph_neighbors"]
    assert index.calls[1][1]["seeds"] == ["E0"] and index.calls[1][1]["exclude_passages"] == ["C0"]
    assert result.evidence_ids == ["C0", "C1"]


def test_tree_browse_observation_exposes_child_scope():
    index = Index()
    client = Client([{"tool": "tree_browse", "node": "T0", "top_k": 2}, tree_action(scope="T2")], stop_after=2)
    research("Find archive records in the relevant subtree.", index, client, {})
    assert [name for name, _ in index.calls] == ["tree_browse", "tree_search"]
    assert index.calls[1][1]["scope"] == "T2"


def test_duplicate_request_is_blocked_before_execution():
    index, client = Index(), Client([tree_action(), tree_action(), {"tool": "answer"}])
    result = research("Find the Cedar Archive acquisition record.", index, client, {})
    assert len(index.calls) == 1
    assert result.trace[1]["blocked_before_execution"] and not result.trace[1]["executed"]
    assert result.termination == "model_sufficient"


def test_graph_ablation_keeps_other_knowledge_tools():
    schema = decision_schema(EvidenceMemory(Index.context, TokenClient()), Index(), allow_graph=False)
    tools = {v["properties"]["tool"]["const"] for v in schema["properties"]["action"]["oneOf"]}
    assert tools == {"tree_search", "tree_browse"}


@pytest.mark.parametrize("dataset", ["math", "longbench"])
@pytest.mark.parametrize("with_evidence", [True, False])
def test_final_adapter_uses_v2_messages_and_one_raw_generation(tmp_path, monkeypatch, dataset, with_evidence):
    from repar.benchmarks import v3_suite
    from repar.benchmarks.math_exam import solver_messages
    from repar.evaluation import official_prompt
    evidence = "[C0] Harbor Library acquired the Cedar Archive in 2022." if with_evidence else ""
    result = ResearchResult(evidence, ["C0"] if with_evidence else [], [], [], "budget", {})
    monkeypatch.setattr(v3_suite, "research", lambda *a, **kw: result)
    class Final(TokenClient):
        config = {}
        def __init__(self): self.calls = []
        def post(self, endpoint, data): return {"prompt": "rendered"}
        def complete(self, messages, budget, **kwargs):
            self.calls.append((messages, budget, kwargs))
            return 'Free-form output with an unfinished {"answer":"fragment"'
    client = Final(); suite = v3_suite.Suite.__new__(v3_suite.Suite)
    suite.client = lambda: client; suite.output = tmp_path
    suite.solver = {"agent_version": "react-v3-core-r1", "context_window": 262144, "answer_max_tokens": 4096}
    suite.config = {"longbench_final_max_tokens": 128, "longbench_answer_grammar": "same-v2-grammar"}
    item = {"question": "When was the archive acquired?", **{f"choice_{c}": c for c in "ABCD"}}
    row = suite.generate("fixture", "original task", Index(), dataset,
                         longbench_item=item if dataset == "longbench" else None)
    assert row["status"] == "ok" and len(client.calls) == 1
    assert row["response"] == 'Free-form output with an unfinished {"answer":"fragment"'
    messages, budget, kwargs = client.calls[0]
    if dataset == "math":
        assert messages == solver_messages("original task", evidence or "未取得相关教材内容。")
        assert budget == 4096 and kwargs == {"tag": "math_repar_answer"}
    else:
        assert messages == [{"role": "user", "content": official_prompt(item, evidence or "No source evidence was obtained.", rag=True)}]
        assert budget == 128 and client.config["answer_grammar"] == "same-v2-grammar"
