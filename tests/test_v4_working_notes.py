from types import SimpleNamespace

import jsonschema
import pytest

from repar.v4.span_feedback import bind_reflection, reflection_request
from repar.v4.reflection_service import reflection_client
from repar.v4.working_notes import working_note
from repar.v4.workflow import research
from test_v4 import Client, Index


def test_working_note_uses_only_sources_in_both_requests():
    def step(assessment, cited, observed):
        return {"reflection": {"assessment": assessment, "supported_sources": cited},
                "reflection_context": {"available_sources": observed}}

    result = SimpleNamespace(evidence_ids=["C4", "C7"], evidence_records=[], trace=[
        step("The transfer took place in spring.", ["C4", "C4"], ["C4"]),
        step("Unseen source.", ["C9"], ["C9"]),
        step("Unseen in this reflection.", ["C7"], ["C4"]),
    ])
    assert working_note(result) == "[C4] The transfer took place in spring."
    result.evidence_ids = []
    assert working_note(result) == ""


def test_span_reference_keeps_the_exact_observed_window_and_final_visibility():
    observation = {"status": "ok", "source_views": [
        {"id": "C1", "start": 7, "end": 22, "text": "deposit arrived"},
        {"id": "C1", "start": 31, "end": 46, "text": "sent to archive"},
    ]}
    context = {"observation": observation, "existing_sources": [], "available_sources": ["C1"]}
    _, schema, windows = reflection_request("Who holds the deposit?", {"objective": "Find the receiver"},
                                             {"name": "read"}, context)
    output = {"support": ["V2"], "assessment": "The archive received the deposit.", "lesson": "Read the handover.",
              "gap": "", "next": "answer"}
    jsonschema.validate(output, schema)
    reflection = bind_reflection(output, windows)
    assert reflection["source_spans"] == [{"id": "C1", "start": 31, "end": 46}]
    result = SimpleNamespace(evidence_ids=["C1"], evidence_records=observation["source_views"][:1],
                             trace=[{"reflection": reflection, "reflection_context": context}])
    assert working_note(result) == ""
    result.evidence_records = observation["source_views"]
    assert working_note(result) == "[C1] The archive received the deposit.\nRead the handover."


def test_exclusive_reflection_requires_a_new_information_need_to_continue():
    feedback = {"observation": {"status": "no_match", "source_views": []}, "existing_sources": []}
    _, schema, _ = reflection_request("Who signed the receipt?", {"objective": "Find signer"},
                                      {"name": "tree_search"}, feedback, exclusive_decision=True)
    base = {"support": [], "assessment": ""}
    jsonschema.validate({**base, "next": "answer"}, schema)
    jsonschema.validate({**base, "next": "continue", "gap": "Name on the signature line"}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**base, "next": "answer", "gap": "Name on the signature line"}, schema)


def test_compact_reflection_binds_supported_decision_to_the_window():
    text = "June ledger held by the Orchard Archive."
    feedback = {"observation": {"status": "ok", "source_views": [
        {"id": "C2", "start": 53, "end": 53 + len(text), "text": text}]}, "existing_sources": []}
    _, schema, windows = reflection_request("Who holds the June ledger?", {"objective": "Find holder"},
                                            {"name": "tree_search"}, feedback, compact=True)
    answer = {"reason": "Holder identified.", "assessment": text, "support": "V1", "next": "answer"}
    jsonschema.validate(answer, schema)
    result = bind_reflection(answer, windows)
    assert result["source_spans"] == [{"id": "C2", "start": 53, "end": 53 + len(text)}]
    look_up = {**answer, "next": "continue", "support": "none", "reason": "Check the transfer date."}
    jsonschema.validate(look_up, schema)
    result = bind_reflection(look_up, windows)
    assert result["source_spans"] == [] and result["gap"] == "Check the transfer date."


def test_reflection_route_keeps_model_sampling_and_seed(tmp_path):
    primary = SimpleNamespace(
        config={"base_url": "http://127.0.0.1:9001", "deployment_id": "evaluation/primary",
                "enable_thinking": False, "model": "test-model", "sampling": {"temperature": 0.1}, "seed": 17},
        base="http://127.0.0.1:9001", cache_dir=tmp_path, call_seconds=15)
    route = {"base_url": "http://127.0.0.1:9002", "deployment_id": "reasoning", "enable_thinking": True}
    secondary = reflection_client(primary, route)
    assert secondary.base == route["base_url"] and secondary.cache_dir == primary.cache_dir
    assert secondary.config["deployment_id"] == "reasoning/evaluation/primary"
    assert secondary.config["enable_thinking"] and not primary.config["enable_thinking"]
    assert all(secondary.config[key] == primary.config[key] for key in ("model", "sampling", "seed"))


def test_alternative_claims_reach_planner_while_explicit_anchor_reaches_search():
    import json

    class AnchoredClient(Client):
        def complete(self, messages, maximum, **kwargs):
            tag = kwargs["tag"]
            data = json.loads(messages[-1]["content"])
            if tag == "v4_plan_1":
                assert "Juniper Hall or Aster Museum" in data["task"]
                return '{"objective":"Read the transfer","query":"Willow move"}'
            if tag == "v4_action_1":
                search = next(item for item in data["available_actions"] if item["name"] == "tree_search")
                assert search["parameters"]["anchor_query"] == "Where did the Willow collection go?"
                return json.dumps({"action_id": search["action_id"]})
            return '{"support":["V1"],"assessment":"The move is recorded.","next":"answer"}'

    index = Index()
    result = research("Did the Willow collection move to Juniper Hall or Aster Museum?", index, AnchoredClient([]),
                      {"agent_version": "react-v4-r43"}, query_anchor="Where did the Willow collection go?")
    assert result.termination == "model_sufficient"
    assert any(params["query"] == "Where did the Willow collection go?" for _, params in index.calls)
