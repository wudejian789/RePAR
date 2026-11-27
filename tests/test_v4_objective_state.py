import json

from repar.v4 import research
from repar.v4.objective_state import ObjectiveState
from test_v4 import Index, TokenClient


def test_partial_coverage_and_findings_survive_reordering_and_revision():
    goals = ObjectiveState()
    goals.apply_plan({"objectives": [
        {"change": "add", "objective": "Find transfer date"},
        {"change": "add", "objective": "Find recipient"},
    ]})
    goals.apply_reflection({"updates": [
        {"objective_id": "O1", "status": "covered", "finding": "Transfer in May", "sources": ["C0"]},
        {"objective_id": "O2", "status": "partial", "finding": "The collection went overseas", "sources": ["C0"], "gap": "Recipient name"},
    ]})
    original = goals.snapshot()
    goals.apply_plan({"objectives": [
        {"change": "keep", "id": "O2"},
        {"change": "revise", "id": "O1", "objective": "Verify day of transfer"},
    ]})
    assert original["active"] == ["O1", "O2"]
    assert original["objectives"][0]["gap"] == ""
    assert goals.active == ["O2", "O1"]
    assert goals.records["O2"] == original["objectives"][1]
    assert goals.records["O1"]["findings"] == [{"text": "Transfer in May", "sources": ["C0"]}]
    assert goals.pending() == ["O2", "O1"]


def test_replanning_does_not_reoffer_covered_goal_and_reflection_can_cover_several_goals():
    class Client(TokenClient):
        def complete(self, messages, maximum, *, tag, **kwargs):
            state = json.loads(messages[1]["content"])
            if tag == "v4_plan_1":
                return json.dumps({"objectives": [
                    {"change": "add", "objective": "Find the acquisition date"},
                    {"change": "add", "objective": "Determine the destination"},
                    {"change": "add", "objective": "Verify when the collection moved"}]})
            if tag == "v4_plan_2":
                assert state["objective_state"]["objectives"][0]["gap"] == ""
                return '{"objectives":[{"change":"keep","id":"O3"},{"change":"keep","id":"O2"}]}'
            if tag == "v4_action_1":
                return '{"objective_id":"O1","tool":"tree_search","parameters":{"query":"Willow acquisition","scope":"T0","top_k":1}}'
            if tag == "v4_action_2":
                assert state["pending_objectives"] == ["O3", "O2"]
                assert state["objective_state"]["objectives"][0]["findings"][0]["sources"] == ["C0"]
                assert state["previous_actions"][0]["gap"] == "Find the acquisition date"
                assert state["previous_actions"][0]["result_sources"] == ["C0"]
                assert kwargs["schema"]["oneOf"][0]["properties"]["objective_id"]["enum"] == ["O3", "O2"]
                return '{"objective_id":"O2","tool":"read","parameters":{"ids":["C0"],"radius":1}}'
            assert state["observation"]["status"] == "ok"
            if tag == "v4_reflection_1":
                assert state["addressed_gap"] == "Find the acquisition date"
                return '{"next":"replan","updates":[{"objective_id":"O1","status":"covered","finding":"Acquired in 2018","sources":["C0"]}]}'
            assert tag == "v4_reflection_2"
            return json.dumps({"next": "continue", "updates": [
                {"objective_id": "O2", "status": "covered", "finding": "Juniper Hall", "sources": ["C1"]},
                {"objective_id": "O3", "status": "covered", "finding": "Moved in 2023", "sources": ["C1"]}]})

    class AcquisitionIndex(Index):
        def execute(self, tool, parameters):
            if tool == "tree_search":
                self.calls.append((tool, parameters))
                return {"operator": tool, "results": [self.record(0)]}
            return super().execute(tool, parameters)

    index = AcquisitionIndex()
    result = research("When did the Willow collection arrive and move, and where did it go?",
                      index, Client(), {"agent_version": "react-v4-r31"})
    assert result.termination == "model_sufficient"
    assert result.trace[1]["control_transition"] == "all_active_objectives_covered"
    assert result.trace[1]["decision"]["gap"] == "Determine the destination"
    assert set(result.evidence_ids) == {"C0", "C1"}
    assert [goal["gap"] for goal in result.state["objective_state"]["objectives"]] == ["", "", ""]


def test_unsourced_completion_is_left_open():
    goals = ObjectiveState()
    goals.apply_plan({"objectives": [{"change": "add", "objective": "Verify recipient"}]})
    unknown = {"objective_id": "O1", "status": "covered", "finding": "A possible name", "sources": []}
    assert goals.apply_reflection({"updates": [unknown]}) == [unknown]
    assert goals.pending() == ["O1"]


def test_quoted_findings_require_text_from_the_cited_visible_window():
    goals = ObjectiveState(quote_evidence=True)
    goals.apply_plan({"objectives": [{"change": "add", "objective": "When did the transfer occur?"}]})
    update = {"objective_id": "O1", "status": "covered", "finding": "The transfer was in May",
              "evidence": [{"source": "C0", "quote": "transfer was in May"}]}
    reflection = {"next": "answer", "updates": [update]}
    payload = {"available_sources": ["C0", "C1"], "existing_sources": [{"id": "C0", "text": "The site opened."}],
               "observation": {"source_views": [{"id": "C1", "text": "The transfer was in May."}]}}
    assert goals.apply_reflection(reflection, payload) == [update]
    assert goals.pending() == ["O1"]
    update["evidence"] = [{"source": "C1", "quote": "transfer  was in May"}]
    assert goals.apply_reflection(reflection, payload) == []
    assert goals.pending() == []
    assert goals.records["O1"]["findings"][0]["evidence"] == update["evidence"]


def test_r32_continues_after_coverage_and_defers_answer_for_invalid_quote():
    class Client(TokenClient):
        def complete(self, messages, maximum, *, tag, **kwargs):
            state = json.loads(messages[1]["content"])
            if tag == "v4_plan_1":
                return '{"objectives":[{"change":"add","objective":"When did the Willow collection move?"}]}'
            if tag == "v4_action_1":
                tree = kwargs["schema"]["oneOf"][0]["properties"]["parameters"]
                assert "top_k" not in tree["required"]
                return '{"objective_id":"O1","tool":"tree_search","parameters":{"query":"Willow move","scope":"T0"}}'
            if tag == "v4_reflection_1":
                return json.dumps({"next": "continue", "updates": [
                    {"objective_id": "O1", "status": "covered", "finding": "Moved in 2023",
                     "evidence": [{"source": "C1", "quote": "moved to Juniper Hall in 2023"}]}]})
            if tag == "v4_action_2":
                assert state["pending_objectives"] == []
                assert kwargs["schema"]["oneOf"][0]["properties"]["objective_id"]["enum"] == ["O1"]
                return '{"objective_id":"O1","tool":"tree_search","parameters":{"query":"Willow acquisition","scope":"T0","top_k":1}}'
            if tag == "v4_reflection_2":
                return json.dumps({"next": "answer", "updates": [
                    {"objective_id": "O1", "status": "covered", "finding": "Moved in 2018",
                     "evidence": [{"source": "C0", "quote": "moved in 2018"}]}]})
            assert tag == "v4_action_3"
            assert state["last_reflection"]["next"] == "continue"
            assert state["last_reflection"]["updates"] == []
            assert state["last_reflection"]["rejected_updates"] == [{"objective_id": "O1", "status": "covered"}]
            return '{"tool":"answer","parameters":{}}'

    result = research("When did the Willow collection move?", Index(), Client(),
                      {"agent_version": "react-v4-r32", "retrieval_top_k": 6})
    first, second, final = result.trace
    assert first["executed"][0]["parameters"]["top_k"] == 6
    assert second["executed"][0]["parameters"]["top_k"] == 1
    assert second["control_transition"] == "answer_deferred_after_rejected_updates"
    assert final["decision"]["tool"] == "answer"
    assert result.termination == "model_sufficient"
    assert result.state["objective_state"]["objectives"][0]["findings"][0]["text"] == "Moved in 2023"
