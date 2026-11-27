"""Independent, task-agnostic checks of v4 execution and source visibility."""
from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from repar.v4.knowledge import KnowledgeAPI, request_identity
from repar.v4.memory import EvidenceMemory, source_boundaries
from repar.v4.workflow import action_candidates, bind_query, ranked_entities, research
from repar.v4.feedback import reflection_context
from repar.v5 import answer as answer_v5


class TokenClient:
    def tokenize(self, text):
        return list(text)


class Index:
    context = "Aster Museum acquired the Willow collection in 2018.\n\nThe collection moved to Juniper Hall in 2023."
    context_hash = "independent-fixture"

    def __init__(self):
        self.calls = []

    def record(self, i):
        boundary = self.context.index("\n\n")+2
        start, end = (0, boundary) if i == 0 else (boundary, len(self.context))
        return {"id": f"C{i}", "start": start, "end": end, "text": self.context[start:end],
                "score": 2-i, "tree_path": ["T0", f"T{i+1}"],
                "entities": [{"id": "E0", "name": "Willow collection", "passage_ids": ["C0", "C1"]}]}

    def execute(self, tool, parameters):
        self.calls.append((tool, deepcopy(parameters)))
        if tool == "tree_browse":
            return {"operator": tool, "children": [{"id": "T1"}]}
        if tool == "graph_entities":
            return {"operator": tool, "entities": self.record(0)["entities"], "results": []}
        if tool == "tree_search":
            if parameters["query"] == "unknown":
                return {"operator": tool, "results": [{**self.record(0), "score": 0}]}
            return {"operator": tool, "results": [self.record(0 if parameters["query"] == "Willow acquisition" else 1)]}
        if tool == "read" and parameters.get("radius") == 0:
            return {"operator": tool, "results": [self.record(int(parameters["ids"][0][1:]))]}
        return {"operator": tool, "results": [self.record(0), self.record(1)]}


class Client(TokenClient):
    def __init__(self, selectors, stop_after=None):
        self.selectors = iter(selectors)
        self.stop_after, self.reflections, self.events = stop_after, 0, []

    def complete(self, messages, maximum, **kwargs):
        tag, value = kwargs["tag"], json.loads(messages[-1]["content"])
        self.events.append((tag, deepcopy(value), deepcopy(kwargs)))
        if "plan" in tag:
            return json.dumps({"objective": "Locate source evidence", "query": "Willow acquisition"})
        if "action" in tag:
            choice = next(self.selectors)(value["available_actions"])
            return json.dumps({"action_id": choice["action_id"]})
        self.reflections += 1
        assert "observation" in value and "source_views" in value["observation"]
        return json.dumps({"assessment": "The visible raw source has been inspected", "gap": "Check the next source",
            "lesson": "Follow the observed source references", "supported_sources": value["available_sources"][:1],
            "next": "answer" if self.stop_after == self.reflections else "continue"})


def select(name, **parameters):
    return lambda actions: next(a for a in actions if a["name"] == name and all(a["parameters"].get(k) == v for k, v in parameters.items()))


@pytest.mark.parametrize(("draft", "lookup"), [
    ("The Aster Museum acquired the collection in 2009.", True),
    ("", True),
    ("2018, as stated in the question.", False),
])
def test_v5_revises_a_draft_using_the_same_source_workflow(tmp_path, draft, lookup):
    index = Index()
    events = tmp_path/"events.jsonl"
    task = ("When did Aster Museum acquire the Willow collection?" if lookup else
            "Aster Museum acquired the Willow collection in 2018. Restate the year.")

    class PriorClient(Client):
        calls = []

        def complete(self, messages, maximum, **kwargs):
            tag = kwargs["tag"]
            if tag == "v5_direct_answer":
                assert not index.calls and maximum == 96
                assert messages[-1]["content"] == task
                return draft
            if tag == "v5_answer":
                user = messages[-1]["content"]
                assert json.loads(user.split("\n\n", 1)[0])["provisional_answer"] == draft
                assert ("acquired the Willow collection in 2018." in user.split("\n\nTask:\n")[0]) == lookup
                return "2018 [C0]" if lookup else "2018"
            state = json.loads(messages[-1]["content"])
            assert state["provisional_answer"] == draft
            if "reflection" in tag:
                assert index.calls and state["observation"]["source_views"][0]["id"] == "C0"
            return super().complete(messages, maximum, **kwargs)

    operation = select("tree_search") if lookup else select("answer")
    result = answer_v5(task, index, PriorClient([operation], stop_after=1),
                       {"agent_version": "react-v5", "direct_answer_max_tokens": 96, "answer_max_tokens": 64},
                       event_path=events)
    assert result["state"]["direct_answer"] == draft
    assert result["evidence_ids"] == (["C0"] if lookup else [])
    assert result["termination"] == "model_sufficient"
    expected = ["direct_answer", "plan", "action"] + (["observation", "reflection"] if lookup else []) + ["finalizing"]
    assert [json.loads(line)["stage"] for line in events.read_text(encoding="utf-8").splitlines()] == expected


def test_r27_hope_binds_the_action_query_and_blocks_its_repeat():
    index = Index()

    class QueryClient(Client):
        def complete(self, messages, maximum, **kwargs):
            tag = kwargs["tag"]
            state = json.loads(messages[-1]["content"])
            if "plan" in tag:
                assert set(kwargs["schema"]["properties"]) == {"objective"}
                return '{"objective":"确认藏品最近一次交接的地点"}'
            if "action" in tag:
                assert set(state["plan"]) == {"objective"}
                assert all("query" not in action["parameters"] for action in state["available_actions"])
                if tag.endswith("_3"):
                    assert state["previous_actions"][-1]["status"] == "blocked"
                    action = select("read", ids=["C1"])(state["available_actions"])
                    return json.dumps({"action_id": action["action_id"]})
                action = next(action for action in state["available_actions"] if
                              action["name"] == "tree_search"
                              and action["parameters"]["scope"] == "T0"
                              and "anchor_query" not in action["parameters"])
                return json.dumps({"action_id": action["action_id"], "query": "馆藏移交"},
                                  ensure_ascii=False)
            return super().complete(messages, maximum, **kwargs)

    result = research("藏品调拨到哪里了？", index, QueryClient([], stop_after=2),
                      {"agent_version": "react-v4-r27-hope"})
    first, blocked, last = result.trace
    assert first["plan"] == {"objective": "确认藏品最近一次交接的地点"}
    assert first["executed"][0]["parameters"]["query"] == "馆藏移交"
    assert blocked["blocked_before_execution"]["reason"] == "Duplicate completed action"
    assert blocked["executed"] == []
    assert "query" not in last["action_proposal"]
    assert [name for name, _ in index.calls] == ["tree_search", "read"]
    assert result.termination == "model_sufficient"


def test_r27_hope2_runs_independent_searches_in_one_batch_and_shares_the_call_budget():
    index = Index()

    class BatchClient(Client):
        def complete(self, messages, maximum, **kwargs):
            tag = kwargs["tag"]
            state = json.loads(messages[-1]["content"])
            if "plan" in tag:
                return '{"objective":"Establish the source and destination of the collection transfer"}'
            if "action" in tag:
                from jsonschema import ValidationError, validate

                assert state["task"] == "Where was the collection acquired and where did it move?"
                assert all("anchor_query" not in a["parameters"] for a in state["available_actions"])
                assert state["remaining_actions"] == {1: 4, 2: 2, 3: 1}[state["round"]]
                search = next(a["action_id"] for a in state["available_actions"]
                              if a["name"] == "tree_search" and a["purpose"].startswith("Locate raw"))
                acquisition = {"action_id": search, "query": "Willow acquisition"}
                handover = {"action_id": search, "query": "Willow handover"}
                if tag.endswith("_1"):
                    return json.dumps({"actions": [acquisition, acquisition, handover]})
                if tag.endswith("_3"):
                    return json.dumps({"actions": [{"action_id": search, "query": "Willow museum transfer"}]})
                answer = {"action_id": "answer"}
                with pytest.raises(ValidationError):
                    validate({"actions": [acquisition, answer]}, kwargs["schema"])
                with pytest.raises(ValidationError):
                    validate({"actions": [acquisition, handover, acquisition]}, kwargs["schema"])
                return json.dumps({"actions": [acquisition, {"action_id": search, "query": "unknown"}]})
            assert len(state["observations"]) == (2 if tag.endswith("_1") else 1)
            assert state["source_view_tokens"] <= state["source_view_budget"] == 3200
            if tag.endswith("_1"):
                assert state["available_sources"] == ["C0", "C1"]
                assert [o["source_views"][0]["id"] for o in state["observations"]] == ["C0", "C1"]
                assert state["blocked_actions"][0]["reason"] == "Duplicate action in batch"
            return json.dumps({"assessment": "Read the observations", "gap": "Check independently",
                               "lesson": "Preserve the reported dates", "supported_sources": [],
                               "next": "continue"})

    result = research("wrapped task", index, BatchClient([]),
                      {"agent_version": "react-v4-r27-hope2", "react_max_steps": 4, "max_actions_per_round": 4},
                      query_task="Where was the collection acquired and where did it move?")
    first, second, third = result.trace
    assert first["blocked_actions"][0]["reason"] == "Duplicate action in batch"
    assert second["blocked_actions"][0]["reason"] == "Duplicate completed action"
    assert len(third["executed"]) == 1
    assert [parameters["query"] for _, parameters in index.calls] == [
        "Willow acquisition", "Willow handover", "unknown", "Willow museum transfer"]


def test_r27_hope3_observes_the_primary_action_before_considering_other_actions():
    index = Index()

    class AlternativeClient(Client):
        def complete(self, messages, maximum, **kwargs):
            tag = kwargs["tag"]
            state = json.loads(messages[-1]["content"])
            if "plan" in tag:
                if tag.endswith("_2"):
                    assert self.reflections == 1
                    assert state["previous_actions"][-1]["progress"]["new_sources"] == ["C0"]
                return '{"objective":"Establish the source and destination of the collection transfer"}'
            if "action" in tag:
                search = next(a["action_id"] for a in state["available_actions"]
                              if a["name"] == "tree_search" and a["purpose"].startswith("Locate raw"))
                return json.dumps({"actions": [
                    {"action_id": search, "query": "Willow acquisition"},
                    {"action_id": search, "query": "Willow handover"},
                ]})
            assert len(index.calls) == (1 if tag.endswith("_1") else 2)
            assert state["observation"]["source_views"][0]["id"] == ("C0" if tag.endswith("_1") else "C1")
            return super().complete(messages, maximum, **kwargs)

    result = research("Where was the collection acquired and where did it move?", index,
                      AlternativeClient([], stop_after=2), {"agent_version": "react-v4-r27-hope3"})
    first, second = result.trace
    assert first["unused_alternatives"][0]["query"] == "Willow handover"
    assert second["blocked_actions"][0]["reason"] == "Duplicate completed action"
    assert second["decision"]["parameters"]["query"] == "Willow handover"
    assert result.termination == "model_sufficient"


def test_r3_revision_composes_browsed_scope_and_reports_actual_progress():
    class BrowsableIndex(Index):
        def execute(self, tool, parameters):
            if tool == 'tree_browse':
                return {'operator': tool, 'children': [
                    {'id': 'T1', 'preview': 'Willow acquisition'}], 'results': []}
            return super().execute(tool, parameters)

    index = BrowsableIndex()
    client = Client([select('tree_browse', node='T0'), select('tree_search', scope='T1')], stop_after=2)
    result = research('Locate the Willow archive', index, client, {'agent_version': 'react-v4-r27'})
    first, second = result.trace
    assert result.termination == 'model_sufficient'
    assert first['reflection_context']['observation_progress']['new_tree_nodes'] == ['T1']
    assert not first['reflection_context']['observation_progress']['new_sources']
    assert 'C0' not in first['reflection_context']['available_sources']
    chosen = next(action for action in second['available_actions']
                  if action['action_id'] == second['decision']['action_id'])
    assert chosen['scope_preview'] == 'Willow acquisition'
    assert second['reflection_context']['observation_progress']['new_sources'] == ['C0']
    assert second['progress']['new_source_references'] == ['C1']
    assert second['reflection_context']['available_sources'] == ['C0']
    assert client.events[-1][1]['observation_progress']['new_source_reference_count'] == 1
    assert result.evidence_records[0]['text'] == index.record(0)['text']


def test_literal_plan_binds_model_phrases_before_search():
    class LiteralPlanClient(Client):
        def complete(self, messages, maximum, **kwargs):
            if 'plan' in kwargs['tag']:
                return json.dumps({'objective': 'Locate relevant material',
                                   'query_phrases': ['star chart', '星图', 'API', '星图']})
            return super().complete(messages, maximum, **kwargs)

    index = Index()
    task = '星图收藏馆的 API 如何查阅？'
    client = LiteralPlanClient([select('tree_search', query='星图 API')], stop_after=1)
    result = research(task, index, client, {'agent_version': 'react-v4-r28'})
    assert index.calls[0][1]['query'] == '星图 API'
    assert result.trace[0]['query_binding'] == {
        'proposed': ['star chart', '星图', 'API', '星图'], 'accepted': ['星图', 'API'],
        'rejected': ['star chart'], 'used_original_task': False}


def test_literal_plan_falls_back_or_uses_observed_terms():
    task = '哪本档案载有馆藏目录？'
    query, binding = bind_query(['catalogue', '...'], task, [])
    assert query == task and binding['used_original_task']
    query, binding = bind_query(['catalogue', '藏书印', 'étude', '藏书印 étude'], task, ['藏书印', 'étude。'])
    assert query == '藏书印 étude' and binding['rejected'] == ['catalogue', '藏书印 étude']


def test_literal_plan_uses_task_text_and_raw_evidence_without_protocol_labels():
    class LiteralPlanClient(Client):
        def complete(self, messages, maximum, **kwargs):
            if 'plan' in kwargs['tag']:
                state = json.loads(messages[-1]['content'])
                assert state['task'] == '藏品移到了哪里？'
                phrases = ['question_stem', 'collection transfer'] if state['round'] == 1 else ['chars', 'Juniper Hall']
                return json.dumps({'objective': 'Find the location', 'query_phrases': phrases})
            return super().complete(messages, maximum, **kwargs)

    index = Index()
    client = LiteralPlanClient([select('tree_search', query='藏品移到了哪里？'),
                                select('tree_search', query='Juniper Hall')], stop_after=2)
    result = research('{"question_stem":"藏品移到了哪里？"}', index, client, {'agent_version': 'react-v4-r28'},
                      query_task='藏品移到了哪里？')
    assert result.trace[0]['query_binding']['used_original_task']
    assert result.trace[1]['query_binding']['rejected'] == ['chars']
    assert index.calls[1][1]['query'] == 'Juniper Hall'


@pytest.mark.parametrize(("task", "proposal", "focus"), [
    ("透水砖为什么要及时清理？", "渗水铺装 堵塞维护", "渗水铺装 堵塞维护"),
    ("Where did the collection move?", "  collection relocation  ", "collection relocation"),
    ("¿Quién recibió la colección?", "destinatario del archivo", "destinatario del archivo"),
    ("BrineGuard 复位的步骤是什么？", "BrineGuard 重置流程", "BrineGuard 重置流程"),
    ("藏品移到了哪里？", "unknown", "unknown"),
    ("How is pressure calculated?", "ΔP", "ΔP"),
    ("How is pressure calculated?", "   ", "How is pressure calculated?"),
])
def test_source_language_plan_passes_rewrites_to_search_with_original_task(task, proposal, focus):
    class RewriteClient(Client):
        def complete(self, messages, maximum, **kwargs):
            if 'plan' in kwargs['tag']:
                state = json.loads(messages[-1]['content'])
                assert state['task'] == task
                return json.dumps({'objective': 'Find source evidence', 'query': proposal})
            return super().complete(messages, maximum, **kwargs)

    index = Index()
    client = RewriteClient([select('tree_search', query=focus)], stop_after=1)
    result = research('Structured task payload', index, client, {'agent_version': 'react-v4-r29'}, query_task=task)
    assert result.trace[0]['plan']['query'] == focus
    expected = [focus, task] if focus != task else [task]
    assert [parameters['query'] for _, parameters in index.calls] == expected
    observation = result.trace[0]['executed'][0]['observation']
    assert [channel['parameters']['query'] for channel in observation['query_channels']] == expected
    assert observation['status'] == 'ok'
    if proposal == "unknown":
        assert observation['query_channels'][0]['ids'] == []
        assert observation['source_view_query_bindings']['C1']['query'] == task


def test_source_language_plan_keeps_task_channel_in_observed_subtree():
    index = Index()
    memory = EvidenceMemory(index.context, TokenClient())
    memory.observe({'operator': 'tree_browse', 'children': [{'id': 'T1'}]})
    task, query = 'Where did the collection move?', 'collection relocation'
    config = {'agent_version': 'react-v4-r29'}
    actions = action_candidates(task, {'query': query}, memory, set(), config, 0)
    scoped = select('tree_search', scope='T1')(actions)
    assert scoped['parameters']['anchor_query'] == task
    KnowledgeAPI(index).execute(scoped['name'], scoped['parameters'])
    assert [(p['query'], p['scope']) for _, p in index.calls] == [(query, 'T1'), (task, 'T1')]
    assert select('graph_entities', query=query)(actions)


def test_source_language_plan_retrieves_terms_absent_from_the_task():
    from repar.index import KnowledgeIndex

    task = 'Which gallery received the coastal charts?'
    focus = 'atlas accession'
    context = 'The museum recorded atlas accession at Juniper Hall in May.\n\nThe gallery received several coastal charts during the spring exhibition.'
    index = KnowledgeIndex(context, {'chunk_words': 10, 'tree_fanout': 2, 'graph_neighbors': 0,
                                    'index_version': 'ordered-extractive-v1'})
    memory = EvidenceMemory(context, TokenClient())
    config = {'agent_version': 'react-v4-r29', 'retrieval_top_k': 2}
    action = select('tree_search')(action_candidates(task, {'query': focus}, memory, set(), config, 0))
    result = KnowledgeAPI(index, r3_retrieval=True).execute(action['name'], action['parameters'])
    assert [c['ids'] for c in result['query_channels']] == [['C0'], ['C1']]
    assert {record['id'] for record in result['results']} == {'C0', 'C1'}
    assert 'atlas accession' in next(record['text'] for record in result['results'] if record['id'] == 'C0')


def test_objectives_survive_an_action_and_can_be_replanned_from_observation():
    class ObjectiveClient(TokenClient):
        stages = []

        def complete(self, messages, maximum, *, tag, **kwargs):
            self.stages.append(tag)
            state = json.loads(messages[-1]['content'])
            if tag == 'v4_plan_1':
                return '{"objectives":["Find the acquisition date","Find the destination"]}'
            if tag == 'v4_plan_3':
                assert state['previous_objectives'] == ['Find the acquisition date', 'Find the destination']
                assert state['previously_covered'] == [0]
                assert state['last_reflection']['next'] == 'replan'
                return '{"objectives":["Verify the destination from its original record"]}'
            if tag == 'v4_action_1':
                assert not state['covered_objectives'] and 'graph_entities' not in {t['tool'] for t in state['tools']}
                return '{"objective_index":0,"tool":"tree_search","parameters":{"query":"Willow acquisition","scope":"T0","top_k":2}}'
            if tag == 'v4_action_2':
                assert state['objectives'][1] == 'Find the destination' and state['covered_objectives'] == [0]
                return '{"objective_index":1,"tool":"tree_browse","parameters":{"node":"T1","top_k":1}}'
            if tag == 'v4_action_3':
                assert state['objectives'] == ['Verify the destination from its original record']
                return '{"objective_index":0,"tool":"read","parameters":{"ids":["C1"],"radius":0}}'
            assert state['observation']['status'] == 'ok'
            result = {'assessment': 'An observation was returned', 'gap': 'Check the destination',
                      'lesson': 'Consult original records', 'supported_sources': state['available_sources'][:1],
                      'covered_objectives': [0],
                      'next': {'v4_reflection_1': 'continue', 'v4_reflection_2': 'replan',
                               'v4_reflection_3': 'answer'}[tag]}
            return json.dumps(result)

    index = Index()
    client = ObjectiveClient()
    result = research('When was Willow acquired and where did it go?', index, client,
                      {'agent_version': 'react-v4-r30'}, allow_graph=False)
    assert client.stages == ['v4_plan_1', 'v4_action_1', 'v4_reflection_1', 'v4_action_2',
                             'v4_reflection_2', 'v4_plan_3', 'v4_action_3', 'v4_reflection_3']
    assert [step['planned'] for step in result.trace] == [True, False, True]
    assert [step['decision']['objective_index'] for step in result.trace] == [0, 1, 0]
    assert index.calls[0][1]['query'] == 'Willow acquisition'
    assert index.calls[0][1]['top_k'] == 2
    assert [item[0] for item in index.calls] == ['tree_search', 'tree_search', 'tree_browse', 'read']
    assert result.termination == 'model_sufficient'


def test_objective_controller_blocks_duplicate_and_unobserved_parameters_before_execution():
    class ObjectiveClient(TokenClient):
        stages = []

        def complete(self, messages, maximum, *, tag, **kwargs):
            self.stages.append(tag)
            if tag == 'v4_plan_1':
                return '{"objectives":["Find a source"]}'
            if tag == 'v4_action_1':
                return '{"objective_index":0,"tool":"tree_search","parameters":{"query":"Willow acquisition","scope":"T99","top_k":2}}'
            if tag in {'v4_action_1_repair', 'v4_action_2'}:
                return '{"objective_index":0,"tool":"tree_search","parameters":{"query":"Willow acquisition","scope":"T0","top_k":2}}'
            if tag == 'v4_action_3':
                state = json.loads(messages[-1]['content'])
                assert state['previous_actions'][-1]['status'] == 'blocked'
                return '{"objective_index":0,"tool":"read","parameters":{"id":"C0","offset":90,"length":10}}'
            assert tag == 'v4_reflection_1'
            return '{"assessment":"Found text","gap":"Check it","lesson":"Read the source","next":"continue","supported_sources":[],"covered_objectives":[]}'

    index = Index()
    client = ObjectiveClient()
    result = research('Find the original Willow record', index, client, {'agent_version': 'react-v4-r30'})
    assert 'v4_action_1_repair' in client.stages
    assert [item[0] for item in index.calls] == ['tree_search', 'tree_search']
    assert [step['blocked_before_execution']['reason'] for step in result.trace[1:]] == [
        'Duplicate completed action', 'Read offset must be inside the selected source range']
    assert not any(step['executed'] or 'reflection' in step for step in result.trace[1:])


def test_entity_navigation_keeps_a_real_read_available_after_a_miss(monkeypatch):
    from repar.v4 import workflow

    monkeypatch.setattr('repar.v4.rerank.SourceReranker', lambda *args, **kwargs: None)

    class NavigationClient(TokenClient):
        def complete(self, messages, maximum, *, tag, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if tag.startswith('v4_plan'):
                query = 'unknown' if tag.endswith('_1') else 'Willow collection'
                return json.dumps({'query': query, 'objective': 'Find original archive holdings'})
            return json.dumps({'assessment': 'Inspected the actual tool result', 'gap': 'Original holdings',
                'lesson': 'Use source records', 'next': 'continue', 'supported_sources': payload['available_sources']})

    def choose(_client, _task, _plan, _reflection, actions, _config, step):
        return next(action for action in actions if action['name'] == ('tree_search', 'graph_entities', 'read')[step]), {}

    monkeypatch.setattr(workflow, 'choose_action', choose)
    config = {'source_reranker': {'path': 'fixture'}, 'source_view_policy': 'scored_neighborhood',
              'read_entity_sources': True, 'native_action_tools': True, 'react_max_steps': 3}
    old = research('Locate archive holdings', Index(), NavigationClient(), {**config, 'agent_version': 'react-v4-r14'})
    assert old.termination == 'no_progress' and old.evidence_ids == []
    new = research('Locate archive holdings', Index(), NavigationClient(), {**config, 'agent_version': 'react-v4-r15'})
    assert new.trace[1]['progress']['new_source_references'] == ['C0', 'C1']
    assert new.trace[1]['reflection_context']['available_sources'] == []
    assert new.trace[2]['executed'][0]['name'] == 'read' and new.evidence_ids == ['C0']


def test_request_identity_uses_full_payload_without_semantic_rules():
    assert request_identity("read", {"b": 2, "a": "same"}) == request_identity("read", {"a": "same", "b": 2})
    assert request_identity("search", {"query": "Willow collection"}) != request_identity("search", {"query": "Willow  collection"})
    assert request_identity("read", {"offset": 10}) != request_identity("read", {"offset": 11})


def test_focus_and_task_channels_are_separate_real_queries():
    index = Index()
    result = KnowledgeAPI(index).execute("tree_search", {"query": "Willow acquisition", "anchor_query": "Where did it move?",
        "scope": "T0", "top_k": 8, "beam_width": 12, "mode": "hybrid"})
    assert [p["query"] for _, p in index.calls] == ["Willow acquisition", "Where did it move?"]
    assert len(result["query_channels"]) == 2
    assert {r["id"] for r in result["results"]} == {"C0", "C1"}


def test_full_disjoint_query_channels_both_can_contribute():
    class FullIndex:
        def execute(self, name, params):
            offset = 0 if params['query'] == 'archive ownership' else 8
            return {'operator': name, 'results': [
                {'id': f'C{i+offset}', 'score': 10-i, 'start': i+offset, 'end': i+offset+1, 'text': 'x'} for i in range(8)]}
    result = KnowledgeAPI(FullIndex()).execute('tree_search', {'query': 'archive ownership',
        'anchor_query': 'museum inventory history', 'top_k': 8})
    ids = {r['id'] for r in result['results']}
    assert len(ids) == 8
    assert ids & {f'C{i}' for i in range(8)}
    assert ids & {f'C{i}' for i in range(8, 16)}
    assert result['fusion']['channel_weights'] == {'focus': 1, 'task': 1}


def test_fusion_rewards_agreement_and_preserves_original_records():
    class OverlapIndex:
        def execute(self, name, params):
            ids = [0, 1, 2, 3] if params['query'] == 'location' else [4, 1, 5, 6]
            return {'operator': name, 'results': [
                {'id': f'C{i}', 'start': i*10, 'end': i*10+5, 'text': f'item{i}', 'score': 2} for i in ids]}
    result = KnowledgeAPI(OverlapIndex()).execute('tree_search', {'query': 'location',
        'anchor_query': 'collection history', 'top_k': 4})
    assert result['results'][0]['id'] == 'C1'
    assert all(r['text'] == f"item{r['id'][1:]}" for r in result['results'])
    assert len({r['id'] for r in result['results']}) == 4


def test_result_limit_is_applied_after_anchored_channel_fusion():
    class OverlapIndex:
        context = 'First. Common. Other.'

        def execute(self, name, params):
            ids = [0, 1] if params['query'] == 'archive custody' else [2, 1]
            spans = [(0, 7), (7, 15), (15, 21)]
            return {'operator': name, 'results': [
                {'id': f'C{i}', 'start': spans[i][0], 'end': spans[i][1],
                 'text': self.context[slice(*spans[i])], 'score': 2}
                for i in ids[:params['top_k']]]}

    index = OverlapIndex()
    parameters = {'query': 'archive custody', 'anchor_query': 'who holds the archive?', 'top_k': 1}
    assert KnowledgeAPI(index).execute('tree_search', parameters)['results'][0]['id'] == 'C0'
    result = KnowledgeAPI(index, candidate_top_k=2, fuse_before_limit=True).execute('tree_search', parameters)
    assert [r['id'] for r in result['results']] == ['C1']
    assert [c['ids'] for c in result['query_channels']] == [['C0', 'C1'], ['C2', 'C1']]
    assert [c['parameters']['top_k'] for c in result['query_channels']] == [2, 2]
    assert result['executed_parameters']['top_k'] == 1
    memory = EvidenceMemory(index.context, TokenClient())
    memory.observe(result)
    feedback = reflection_context(memory, result, parameters['query'], 100, 'search')
    assert [c['ids'] for c in feedback['observation']['query_channels']] == [['C1'], ['C1']]
    assert feedback['available_sources'] == ['C1']


def test_independent_query_limits_keep_both_requested_source_hits():
    index = Index()
    api = KnowledgeAPI(index, fuse_before_limit=True, use_channel_view_queries=True, limit_each_search_channel=True)
    params = {'query': 'Willow acquisition', 'anchor_query': 'Where was the collection moved?', 'top_k': 1}
    result = api.execute('tree_search', params)
    assert result['fusion']['result_limit_scope'] == 'each_query'
    assert [r['id'] for r in result['results']] == ['C0', 'C1']
    assert result['source_view_query_bindings']['C1']['query'] == params['anchor_query']
    assert all(p['top_k'] == 1 for _, p in index.calls)
    same = api.execute('tree_search', {**params, 'anchor_query': params['query']})
    assert [r['id'] for r in same['results']] == ['C0']
    assert len(index.calls) == 2


def test_source_view_uses_its_best_retrieval_channel_and_survives_query_changes():
    context = ('Port staff moved three boxes. Staff stored receipts in a drawer. The box count was confirmed. '
               'The centre opens at six. Parking spaces are limited. A map is available online. '
               'Visitors must request an escort to inspect the archive. Inspection slots are limited.')
    focus = 'moved boxes'
    anchor = 'archive escort'

    class SourceIndex:
        def execute(self, name, params):
            return {'results': [{'id': 'C0', 'text': context, 'start': 0, 'end': len(context), 'score': 1}]
                    if params['query'] == anchor else []}

    api = KnowledgeAPI(SourceIndex(), use_channel_view_queries=True)
    observation = api.execute('tree_search', {'query': focus, 'anchor_query': anchor, 'top_k': 1})
    assert observation['source_view_query_bindings'] == {'C0': {'rank': 0, 'channel': 'task', 'query': anchor}}
    memory = EvidenceMemory(context, TokenClient(), 200, source_view_policy='scored_neighborhood')
    memory.observe(observation)
    payload = reflection_context(memory, observation, focus, 200, 'retrieval')
    visible = payload['observation']['source_views'][0]
    assert 'request an escort' in visible['text']
    evidence, views = memory.render('new focus on the receipt shelves', 200)
    assert 'request an escort' in evidence
    assert views[0]['text'] == visible['text'] == context[visible['start']:visible['end']]


def test_identical_base_channel_reused_between_single_and_dual_queries():
    index, api = Index(), KnowledgeAPI(Index())
    parameters = {'query': 'Willow acquisition', 'scope': 'T0', 'top_k': 8, 'beam_width': 12, 'mode': 'hybrid'}
    one = api.execute('tree_search', parameters)
    two = api.execute('tree_search', {**parameters, 'anchor_query': 'Where did it move?'})
    assert len(api.index.calls) == 2
    assert [x['cached'] for x in two['query_execution']] == [True, False]
    assert one['results'][0]['id'] in {r['id'] for r in two['results']}
    assert all(c['parameters']['query'] in ('Willow acquisition', 'Where did it move?') for c in api.search_events)


def test_query_cache_is_session_local_and_returns_detached_records():
    index = Index(); api = KnowledgeAPI(index)
    parameters = {'query': 'Willow acquisition', 'top_k': 8}
    one = api.execute('tree_search', parameters)
    one['results'][0]['text'] = 'modified caller data'
    two = api.execute('tree_search', parameters)
    assert len(index.calls) == 1 and two['results'][0]['text'] != 'modified caller data'
    another = KnowledgeAPI(index)
    another.execute('tree_search', parameters)
    assert len(index.calls) == 2


def test_query_cache_does_not_reuse_different_scope_or_parameters():
    api = KnowledgeAPI(Index())
    for p in [{'query': 'Willow acquisition', 'scope': 'T0', 'top_k': 8},
              {'query': 'Willow acquisition', 'scope': 'T1', 'top_k': 8},
              {'query': 'Willow acquisition', 'scope': 'T0', 'top_k': 4}]:
        api.execute('tree_search', p)
    assert len(api.index.calls) == 3


def test_zero_match_and_graph_exclusion_are_strict():
    api = KnowledgeAPI(Index())
    assert api.execute("tree_search", {"query": "unknown", "top_k": 8})["results"] == []
    assert [r["id"] for r in api.execute("graph_neighbors", {"exclude_passages": ["C0"]})["results"]] == ["C1"]


def test_completed_requests_removed_before_model_choice():
    memory = EvidenceMemory(Index.context, TokenClient())
    plan = {"query": "Willow acquisition"}
    before = action_candidates("Find acquisition", plan, memory, set(), {}, 0)
    call = next(a for a in before if a["name"] == "tree_search")
    after = action_candidates("Find acquisition", plan, memory, {request_identity(call["name"], call["parameters"])}, {}, 1)
    assert call["parameters"] not in [a["parameters"] for a in after]
    assert {a["name"] for a in after} >= {"tree_search", "graph_entities", "tree_browse"}


def test_no_prescribed_tree_graph_route():
    memory = EvidenceMemory(Index.context, TokenClient())
    actions = action_candidates("Find archive history", {"query": "Willow acquisition"}, memory, set(), {}, 0)
    assert {a["name"] for a in actions} == {"tree_search", "graph_entities", "tree_browse"}
    without_graph = action_candidates("Find history", {"query": "Willow acquisition"}, memory, set(), {}, 0, False)
    assert not any(a["name"].startswith("graph_") for a in without_graph)


def test_joint_plan_validates_known_references_and_binds_the_generated_query():
    import jsonschema
    from repar.v4.planning import bind_plan, plan_request

    memory = EvidenceMemory(Index.context, TokenClient())
    config = {'agent_version': 'react-v4-r23', 'retrieval_budget_options': [1, 8]}
    candidates = action_candidates('Locate the archives', {'query': 'previous focus'}, memory, set(), config, 0)
    _, schema = plan_request({'task': 'Locate the archives'}, candidates, [1, 8])
    value = {'objective': 'Find the custodian', 'query': 'archive custody',
             'action': {'name': 'tree_search', 'scope': 'T0', 'top_k': 1, 'include_task': True}}
    jsonschema.validate(value, schema)
    plan, call = bind_plan(value, candidates, 'Locate the archives')
    assert plan['query'] == call['parameters']['query'] == 'archive custody'
    assert call['parameters']['anchor_query'] == 'Locate the archives'
    assert call['parameters']['top_k'] == 1
    value['action']['scope'] = 'T99'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)


def test_joint_planner_reselects_a_duplicate_before_spending_a_tool_step(monkeypatch):
    from repar.v4 import workflow
    from repar.v4 import planning

    calls = {'joint': 0, 'fallback': 0}
    query = 'Willow acquisition'
    parameters = {'query': query, 'scope': 'T0', 'top_k': 1, 'beam_width': 12, 'mode': 'hybrid'}

    def selected_plan(client, state, candidates, limits, budget, tag):
        calls['joint'] += 1
        return {'objective': 'Locate original custody evidence', 'query': query}, {
            'name': 'tree_search', 'parameters': dict(parameters), 'action_id': 'a0', 'purpose': 'Find source'}

    def choose(_client, _task, _plan, _reflection, actions, _config, _step):
        calls['fallback'] += 1
        assert not any(action['name'] == 'tree_search' and action['parameters'] == parameters for action in actions)
        return next(action for action in actions if action['name'] == 'tree_browse'), {}

    class Reflections(Client):
        def complete(self, messages, maximum, **kwargs):
            tag, payload = kwargs['tag'], json.loads(messages[-1]['content'])
            if 'reflection' in tag:
                return json.dumps({'assessment': 'Inspected the returned original source', 'gap': 'Find context',
                    'lesson': 'Continue locating the source', 'next': 'continue', 'supported_sources': payload['available_sources']})
            return super().complete(messages, maximum, **kwargs)

    monkeypatch.setattr(planning, 'select_plan', selected_plan)
    monkeypatch.setattr(workflow, 'choose_action', choose)
    config = {'agent_version': 'react-v4-r24', 'native_action_tools': False, 'read_entity_sources': True,
        'react_max_steps': 2, 'no_progress_patience': 2}
    result = research('Find original custody details', Index(), Reflections([], stop_after=None), config)
    assert calls == {'joint': 2, 'fallback': 1}
    assert result.trace[1]['joint_selected']['parameters'] == parameters
    assert result.trace[1]['duplicate_reselection']['rejected'] == result.trace[1]['joint_selected']
    assert result.trace[1]['decision']['tool'] == 'tree_browse'
    assert result.trace[1]['executed'][0]['name'] == 'tree_browse'
    assert not result.trace[1].get('blocked_before_execution')


def test_browsed_scope_is_immediately_searchable_with_bound_mode():
    import jsonschema
    from repar.v4.planning import bind_plan, plan_request

    memory = EvidenceMemory(Index.context, TokenClient())
    memory.observe({'operator': 'tree_browse', 'node': {'id': 'T0'},
                    'children': [{'id': 'T1', 'preview': 'Shipping receipts'},
                                 {'id': 'T2', 'preview': 'Meeting minutes'},
                                 {'id': 'T3', 'preview': 'Preservation inventory'}]})
    task, query = 'Find the inventory', 'Preservation inventory'
    config = {'agent_version': 'react-v4-r25', 'retrieval_budget_options': [1, 8]}
    actions = action_candidates(task, {'query': query}, memory, set(), config, 0, allow_graph=False)
    messages, schema = plan_request({'task': task}, actions, [1, 8], bind_search_scope=True)
    scopes = json.loads(messages[-1]['content'])['knowledge_operations']['tree_search']['scopes']
    assert {'scope': 'T3', 'mode': 'tree', 'observed_preview': 'Preservation inventory'} in scopes
    assert not memory.records and not memory.visible
    value = {'objective': 'Locate the inventory', 'query': query,
             'action': {'name': 'tree_search', 'scope': 'T3', 'top_k': 1}}
    jsonschema.validate(value, schema)
    _, call = bind_plan(value, actions, task)
    assert call['parameters']['scope'] == 'T3' and call['parameters']['mode'] == 'tree'
    value['action']['include_task'] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)


def test_focused_read_offers_an_unseen_original_range_from_a_known_source():
    part = 'Routine desk schedules and linen supplies. '
    target = 'The report amendment deadline is September 19. '
    context = part + target + part
    memory = EvidenceMemory(context, TokenClient())
    record = {'id': 'C0', 'start': 0, 'end': len(context), 'text': context}
    memory.observe({'results': [record]})
    start_of_last_sentence = len(part) + len(target)
    memory.mark_visible([memory.page(record, start_of_last_sentence, len(context), 200)], 'prior')
    config = {'agent_version': 'react-v4-r26', 'read_page_chars': 100}
    actions = action_candidates('Find the date', {'query': 'amendment deadline'}, memory, set(), config, 0)
    reads = [action for action in actions if action['name'] == 'read' and 'offset' in action['parameters']]
    assert [action['parameters']['offset'] for action in reads] == [0, len(part)]
    params = reads[1]['parameters']
    assert context[params['offset']:params['offset'] + params['length']] == target
    assert 'September' not in str(reads[1])


def test_entity_candidates_rank_observed_entities_beyond_insertion_prefix():
    memory = EvidenceMemory(Index.context, TokenClient())
    memory.entities = {f"E{i}": {"id": f"E{i}", "name": "unrelated", "passage_ids": ["C9"]} for i in range(40)}
    memory.entities["E40"] = {"id": "E40", "name": "Willow collection", "passage_ids": ["C0", "C1"]}
    assert ranked_entities(memory, "Willow collection")[0]["id"] == "E40"
    actions = action_candidates("Willow history", {"query": "Willow collection"}, memory, set(), {}, 0)
    assert any(a["name"] == "graph_neighbors" and a["parameters"]["seeds"] == ["E40"] for a in actions)


def test_visible_span_ledger_counts_partial_reads_on_same_source():
    index = Index()
    memory = EvidenceMemory(index.context, TokenClient())
    memory.observe({"results": [index.record(0)]})
    a = memory.page(index.record(0), 0, 20, 10)
    b = memory.page(index.record(0), a["end"], 30, 10)
    assert memory.mark_visible([a], "first") > 0
    assert memory.mark_visible([a], "repeat") == 0
    assert memory.mark_visible([b], "continue") > 0
    assert memory.unread("C0") == b["end"]
    assert all(index.context[r["start"]:r["end"]] == r["text"] for r in (a, b))


def test_cursor_bounds_and_exact_source_text():
    index = Index()
    api = KnowledgeAPI(index)
    result = api.execute("read", {"id": "C0", "offset": 10, "length": 12})
    memory = EvidenceMemory(index.context, TokenClient())
    memory.observe(result)
    view = memory.observation_view(result, "acquisition", 200, "read")["source_views"][0]
    assert view["start"] == 10 and view["end"] == 22
    assert view["truncated_before"] and view["truncated_after"]
    with pytest.raises(ValueError, match="cursor"):
        api.execute("read", {"id": "C0", "offset": 500, "length": 20})
    with pytest.raises(ValueError, match="offsets"):
        memory.observe({"results": [{"id": "C9", "start": 0, "end": 4, "text": "fake"}]})


def test_packing_preserves_multiple_sources_and_dynamic_source_ranges():
    index = Index()
    memory = EvidenceMemory(index.context, TokenClient(), 170)
    memory.observe({"results": [index.record(0), index.record(1)]})
    rendered, views = memory.render("Willow collection acquired moved", stage="controller")
    assert len(rendered) <= 170
    assert {r["id"] for r in views} == {"C0", "C1"}
    assert all(index.context[r["start"]:r["end"]] == r["text"] for r in views)
    assert memory.visible_characters > 0


def test_unspaced_long_source_uses_token_budget_and_allows_multiple_windows():
    context = "观测站保存了原始仪器记录。"*100+"档案馆后来接收这些记录。"*100
    memory = EvidenceMemory(context, TokenClient(), 400)
    memory.observe({"results": [{"id": "C0", "start": 0, "end": len(context), "text": context}]})
    rendered, views = memory.render("档案馆记录", stage="controller")
    assert views and len(rendered) <= 400
    assert all(context[r["start"]:r["end"]] == r["text"] for r in views)
    assert memory.unread("C0") is not None


def test_automatic_window_keeps_selected_sentence_and_complete_preceding_context():
    first = 'Unrelated background about catalog maintenance. '
    condition = 'The following permission applies only while the lease is active. '
    selected = 'Amber archive insurance audit confirms inventory ownership. '
    context = first*7+condition+selected+'Routine storage notes follow. '*15
    memory = EvidenceMemory(context, TokenClient())
    record = {'id': 'C0', 'start': 0, 'end': len(context), 'text': context}
    view = memory._best(record, 'amber archive insurance audit inventory ownership', 180)
    assert selected in view['text'] and condition in view['text']
    assert view['start'] in source_boundaries(context) and view['end'] in source_boundaries(context)
    assert not view['partial_sentence']
    assert view['text'] == context[view['start']:view['end']]


def test_oversized_sentence_budget_fallback_keeps_whole_words_and_marks_partial():
    context = 'Inventory ownership insurance documentation '+('unusuallylongword '*40)+'. '
    memory = EvidenceMemory(context, TokenClient())
    record = {'id': 'C0', 'start': 0, 'end': len(context), 'text': context}
    view = memory._best(record, 'inventory ownership insurance', 100)
    assert 0 < len(view['text']) <= 100 and view['partial_sentence']
    assert 'partial_sentence=true' in memory.block(view)
    assert view['end'] == len(context) or not (context[view['end']-1].isalnum() and context[view['end']].isalnum())


def test_boundaries_keep_decimals_and_wrapped_sentence_together():
    text = 'A reading of 3.14 units was\nrecorded in the report. Another reading follows.\n\nNew section.'
    boundaries = source_boundaries(text)
    assert text.index('14') not in boundaries
    assert text.index('recorded') not in boundaries
    assert text.index('Another') in boundaries and text.index('New section') in boundaries


def test_later_window_does_not_overlap_previously_packed_sentence():
    context = 'Amber records establish archive ownership. Birch records establish storage location. Cedar records establish lease duration.'
    memory = EvidenceMemory(context, TokenClient())
    record = {'id': 'C0', 'start': 0, 'end': len(context), 'text': context}
    first = memory._best(record, 'amber archive ownership', 60)
    second = memory._best(record, 'cedar lease duration', 60, [(first['start'], first['end'])])
    assert second and (second['start'] >= first['end'] or second['end'] <= first['start'])
    assert second['start'] in source_boundaries(context)


def test_automatic_window_keeps_absolute_offsets_for_embedded_source():
    text = 'Unrelated introductory sentence. Amber inventory policy has been approved. Further record keeping details. '
    prefix = 'A separate document appears before this source.\n\n'
    context = prefix+text+'Another source follows.'
    memory = EvidenceMemory(context, TokenClient())
    record = {'id': 'C7', 'start': len(prefix), 'end': len(prefix)+len(text), 'text': text}
    view = memory._best(record, 'amber inventory policy', 60)
    assert view['text'] == context[view['start']:view['end']]
    assert view['start'] >= len(prefix) and view['end'] <= record['end']
    assert view['start']-len(prefix) in source_boundaries(text)
    assert not view['partial_sentence']


def test_current_core_rejects_historical_version_label():
    with pytest.raises(ValueError, match='frozen snapshot'):
        research('Trace archive custody', Index(), Client([]), {'agent_version': 'react-v4'})


def test_entity_only_reflection_receives_retained_text_in_same_call():
    index, memory = Index(), EvidenceMemory(Index.context, TokenClient())
    memory.observe({'results': [index.record(0)]})
    feedback = reflection_context(memory, {'operator': 'graph_entities', 'results': [],
        'entities': index.record(0)['entities'], 'status': 'ok'}, 'Willow acquisition', 300, 'entity')
    assert feedback['observation']['source_views'] == []
    assert feedback['existing_sources'][0]['text'] == index.record(0)['text']
    assert feedback['available_sources'] == ['C0']
    assert feedback['source_view_tokens'] <= 300


def test_reflection_citation_schema_excludes_historical_ids_with_no_current_text():
    index, memory = Index(), EvidenceMemory(Index.context, TokenClient())
    memory.observe({'results': [index.record(0)]})
    memory.visible['C999'] = [(0, 3)]
    current = {'operator': 'tree_search', 'results': [index.record(1)], 'status': 'ok'}
    memory.observe(current)
    feedback = reflection_context(memory, current, 'Willow archive move', 300, 'action')
    text_ids = {v['id'] for v in feedback['observation']['source_views']+feedback['existing_sources']}
    assert set(feedback['available_sources']) == text_ids
    assert 'C999' not in feedback['available_sources']


def test_current_and_retained_views_share_one_budget_and_original_spans():
    index, memory = Index(), EvidenceMemory(Index.context, TokenClient())
    memory.observe({'results': [index.record(0), index.record(1)]})
    feedback = reflection_context(memory, {'operator': 'read', 'results': [index.record(1)]}, 'Willow collection', 200, 'read')
    views = feedback['observation']['source_views']+feedback['existing_sources']
    assert feedback['source_view_tokens'] <= feedback['source_view_budget'] == 200
    assert all(index.context[v['start']:v['end']] == v['text'] for v in views)


def test_reflection_retains_disjoint_prior_range_of_the_same_source():
    first = 'The atlas is held by the East Office. '
    middle = 'An unrelated map was digitized. '
    last = 'Reading the original atlas requires a permit.'
    context = first+middle+last
    record = {'id': 'C0', 'start': 0, 'end': len(context), 'text': context}
    memory = EvidenceMemory(context, TokenClient(), 300, source_view_policy='scored_neighborhood')
    memory.observe({'results': [record]})

    def read_range(lo, hi):
        return {'operator': 'read', 'status': 'ok', 'results': [record],
                'requested_view': {'id': 'C0', 'start': lo, 'end': hi}}

    start = read_range(0, len(first))
    refresh = read_range(len(first+middle), len(context))
    reflection_context(memory, start, 'atlas custody', 300, 'first')
    assert reflection_context(memory, refresh, 'atlas permits', 300, 'old')['existing_sources'] == []
    feedback = reflection_context(memory, refresh, 'atlas permits', 300, 'new', retain_prior_ranges=True)
    assert [r['text'] for r in feedback['existing_sources']] == [first]
    assert [r['text'] for r in feedback['observation']['source_views']] == [last]
    assert feedback['available_sources'] == ['C0']
    assert feedback['source_view_tokens'] <= 300


def test_graph_proposition_is_citable_only_when_matching_original_known_source():
    index, memory = Index(), EvidenceMemory(Index.context, TokenClient())
    memory.observe({'results': [index.record(0)]})
    feedback = reflection_context(memory, {'operator': 'graph_neighbors', 'results': [], 'propositions': [
        {'passage_id': 'C0', 'text': index.record(0)['text']}, {'passage_id': 'C999', 'text': 'Invented claim'}]},
        'Willow archive', 20, 'graph')
    assert feedback['available_sources'] == ['C0']
    assert any(e['stage'] == 'graph_propositions' for e in memory.exposures)


def test_actual_workflow_retains_raw_sources_across_entity_query():
    index = Index(); client = Client([select('tree_search'), select('graph_entities')], stop_after=2)
    result = research('Trace the Willow collection history', index, client, {})
    reflection = next(body for tag, body, _ in client.events if tag == 'v4_reflection_2')
    assert reflection['observation']['source_views'] == []
    assert reflection['existing_sources'] and reflection['available_sources'] == ['C0']
    schema = next(kwargs['schema'] for tag, _, kwargs in client.events if tag == 'v4_reflection_2')
    assert schema['properties']['supported_sources']['items']['enum'] == ['C0']
    assert result.trace[1]['reflection_context']['existing_sources'] == reflection['existing_sources']


def test_actual_observation_precedes_reflection_and_graph_is_optional(tmp_path):
    index, client = Index(), Client([select("tree_search")], stop_after=1)
    result = research("Find the acquisition history", index, client, {}, event_path=tmp_path/"events.jsonl")
    assert [tag for tag, _, _ in client.events] == ["v4_plan_1", "v4_action_1", "v4_reflection_1"]
    assert result.termination == "model_sufficient" and result.evidence_ids == ["C0"]
    observed = client.events[-1][1]["observation"]["source_views"][0]
    assert observed["text"] == index.context[observed["start"]:observed["end"]]
    assert [json.loads(s)["stage"] for s in (tmp_path/"events.jsonl").read_text().split('\n') if s] == ["observation", "reflection"]


def test_repeated_plan_cannot_repeat_completed_execution():
    index = Index()
    def choose_uncompleted(actions):
        return next(a for a in actions if a["name"] == "tree_search")
    client = Client([choose_uncompleted, choose_uncompleted, select("answer")])
    result = research("Find collection sources", index, client, {})
    ids = [request_identity(e["name"], e["parameters"]) for s in result.trace for e in s["executed"]]
    assert len(ids) == len(set(ids)) == 2
    assert result.termination == "model_sufficient"


@pytest.mark.parametrize("dataset", ["math", "longbench"])
def test_v4_adapter_preserves_final_generation_and_no_extra_stage(monkeypatch, tmp_path, dataset):
    from repar.benchmarks import v4_suite
    from repar.benchmarks.math_exam import solver_messages
    from repar.evaluation import official_prompt
    evidence = "[C0] Aster Museum acquired the collection."
    monkeypatch.setattr(v4_suite, "research", lambda *a, **kw: SimpleNamespace(
        evidence=evidence, evidence_ids=["C0"], evidence_records=[], trace=[], state={}, termination="budget"))
    class Final(TokenClient):
        config = {}
        def __init__(self): self.calls = []
        def post(self, route, body): return {"prompt": "rendered"}
        def complete(self, messages, maximum, **kw):
            self.calls.append((messages, maximum, kw)); return 'raw unfinished {"answer"'
    suite = v4_suite.V4Suite.__new__(v4_suite.V4Suite)
    client = Final(); suite.client = lambda: client; suite.output = tmp_path
    suite.solver = {"agent_version": "react-v4", "context_window": 262144, "answer_max_tokens": 4096}
    suite.config = {"longbench_final_max_tokens": 128, "longbench_answer_grammar": "unchanged"}
    item = {"question": "When?", **{f"choice_{c}": c for c in "ABCD"}}
    row = suite.generate("fixture", "task", Index(), dataset, longbench_item=item if dataset == "longbench" else None)
    assert row["status"] == "ok" and row["response"] == 'raw unfinished {"answer"'
    assert len(client.calls) == 1
    messages, maximum, _ = client.calls[0]
    assert maximum == (4096 if dataset == "math" else 128)
    assert messages == (solver_messages("task", evidence) if dataset == "math" else [{"role": "user", "content": official_prompt(item, evidence, rag=True)}])


def test_v4_configuration_retains_model_and_source_identity():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/"configs/repar_v4.json").read_text(encoding="utf-8"))
    old = json.loads((root/"configs/repar_v3.json").read_text(encoding="utf-8"))
    for key in ("base_url", "model", "deployment_id", "context_window", "temperature", "seed", "enable_thinking",
                "answer_max_tokens", "chunk_words", "tree_fanout", "graph_neighbors", "index_version", "graph_model"):
        assert config[key] == old[key]


def test_v4_research_adapter_preserves_structured_task_conditions(monkeypatch, tmp_path):
    from repar.benchmarks import v4_suite
    from repar.benchmarks.math_exam import task_text

    item = {"question_stem": "Determine the parcel's displacement.", "question_type": "single_choice",
            "choices": [{"tag": "A", "text": "five meters"}],
            "figures": [{"description": "The arrow shows the direction of travel.", "unit": "m",
                         "vectors": [{"label": "Travel", "start": [2, 3], "end": [5, 7]}]}]}
    suite = v4_suite.V4Suite.__new__(v4_suite.V4Suite)
    suite.solver = {"agent_version": "react-v4-r52"}
    suite.output = tmp_path

    def capture(task, index, client, config, *, query_task, query_anchor, **kwargs):
        assert task == query_task
        assert item["question_stem"] in task
        assert "A. five meters" in task
        assert "The arrow shows the direction of travel." in task
        assert "Travel：起点 (2, 3)，终点 (5, 7)" in task
        assert query_anchor == item["question_stem"] + "\nfive meters"
        return True

    monkeypatch.setattr(v4_suite, "research", capture)
    assert suite.research_result(task_text(item), Index(), TokenClient(), "math", "independent")


def test_observed_window_survives_query_change_without_reextraction():
    intro = 'The archive collection transfer register lists the archive collection transfer process and collection forms. '
    fact = 'Orchid was transferred from Larch Museum to Elm Repository after the loan expired. '
    text = intro*4+'\n\n'+fact*4
    memory = EvidenceMemory(text, TokenClient(), 230)
    memory.observe({'results': [{'id': 'C0', 'start': 0, 'end': len(text), 'text': text}]})
    _, displayed = memory.discover('Orchid Larch Elm', 230, stage='observation')
    _, final = memory.render('Which collection was transferred to the archive?', 230)
    assert final and displayed
    assert [(r['start'], r['end']) for r in final] == [(r['start'], r['end']) for r in displayed]
    assert all(text[r['start']:r['end']] == r['text'] for r in final)


def test_search_tokenizer_ranks_unread_windows_without_changing_offsets():
    text = '红松正在归档。红松桥今日关闭。'
    record = {'id': 'C0', 'start': 2, 'end': 2 + len(text), 'text': text}

    def search_terms(value):
        return re.findall('红松桥|红松|归档|今日|关闭', value)

    context = '目录' + text
    memory = EvidenceMemory(context, TokenClient(), tokenize_terms=search_terms)
    memory.observe({'results': [record]})

    start, end = memory.focused_unread('C0', '红松桥')
    assert (start, end) == (2 + len('红松正在归档。'), record['end'])
    assert memory.page(record, start, end, 100)['text'] == '红松桥今日关闭。'


def test_single_query_focus_survives_a_later_source_read():
    index = Index()
    api = KnowledgeAPI(index, r3_retrieval=True, use_channel_view_queries=True,
                       bind_single_search_views=True)
    memory = EvidenceMemory(index.context, TokenClient(), retain_search_focus=True)
    observation = api.execute('tree_search', {'query': 'Willow acquisition'})
    memory.observe(observation)
    assert observation['source_view_query_bindings']['C0']['query'] == 'Willow acquisition'

    observation = api.execute('read', {'ids': ['C0'], 'radius': 0})
    memory.observe(observation)
    assert memory.source_view_queries == {'C0': 'Willow acquisition'}
    assert memory.records['C0']['text'] == index.record(0)['text']


def test_corpus_term_weights_select_the_distinctive_unread_sentence():
    from repar.retrieval.lexical import bm25_idf

    text = 'Routine form update. River gauge calibration.'
    record = {'id': 'C0', 'start': 0, 'end': len(text), 'text': text}
    frequency = {'routine': 90, 'calibration': 2}
    memory = EvidenceMemory(text, TokenClient(), term_weight=lambda term: bm25_idf(100, frequency[term]))
    memory.observe({'results': [record]})

    start, end = memory.focused_unread('C0', 'routine calibration')
    assert text[start:end] == 'River gauge calibration.'


def test_final_evidence_order_preserves_ranges_and_prioritizes_matching_passages():
    index = Index()
    memory = EvidenceMemory(index.context, TokenClient())
    memory.observe({'results': [index.record(0), index.record(1)]})

    source_order, original = memory.discover('Juniper Hall')
    ranked, views = memory.discover('Juniper Hall', relevance_order=True)

    assert source_order.startswith('[C0 ')
    assert ranked.startswith('[C1 ')
    assert views == list(reversed(original))
    assert all(index.context[r['start']:r['end']] == r['text'] for r in views)


@pytest.mark.parametrize("needs_source", [False, True])
def test_action_can_use_task_premises_or_actual_prior_evidence(needs_source, monkeypatch):
    from repar.v4 import selection
    from repar.v3.workflow import structured

    class EvidenceClient(Client):
        def complete(self, messages, maximum, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if kwargs['tag'] == 'v4_action_1':
                assert payload['source_evidence'] == ''
                assert any(action['action_id'] == 'answer' for action in payload['available_actions'])
                if not needs_source:
                    return '{"action_id":"answer"}'
            if kwargs['tag'] == 'v4_reflection_1':
                return '{"support":[],"assessment":"","next":"continue","gap":"Confirm the custodian."}'
            if kwargs['tag'] == 'v4_action_2':
                assert index.record(0)['text'] in payload['source_evidence']
                return '{"action_id":"answer"}'
            return super().complete(messages, maximum, **kwargs)

    monkeypatch.setattr(selection, 'structured', structured)
    index = Index()
    task = 'Who received Willow?' if needs_source else 'The museum received Willow. Who received Willow?'
    result = research(task, index, EvidenceClient([select('tree_search')]), {'agent_version': 'react-v4-r54'})
    assert result.termination == 'model_sufficient'
    assert [entry['name'] for step in result.trace for entry in step['executed']] == (
        ['tree_search'] if needs_source else [])
    assert bool(index.calls) == needs_source
    assert len(result.trace) == (2 if needs_source else 1)


def test_stable_view_budget_drops_a_whole_range_and_records_omission():
    text = 'Birch Library keeps the register. Cedar Museum owns the painting. '
    memory = EvidenceMemory(text, TokenClient(), 25)
    record = {'id': 'C0', 'start': 0, 'end': len(text), 'text': text}
    memory.observe({'results': [record]})
    view = memory.page(record, 0, len(text), 100)
    memory.mark_visible([view], 'read')
    evidence, final = memory.render('Who owns the painting?', stage='controller')
    assert evidence == '' and final == []
    assert memory.selections[-1]['dropped_for_budget'] == [{'id': 'C0', 'start': 0, 'end': len(text)}]
    assert memory.selections[-1]['tokens'] == 0 and memory.selections[-1]['selected_ranges'] == []
    assert memory.source_views[('C0', 0, len(text))]['text'] == text


def test_union_of_observed_views_does_not_fill_unseen_holes():
    text = 'A'*100
    memory = EvidenceMemory(text, TokenClient(), 500)
    record = {'id': 'C0', 'start': 0, 'end': 100, 'text': text}
    memory.observe({'results': [record]})
    views = [memory.page(record, a, b, 100) for a, b in [(0, 20), (10, 35), (50, 70)]]
    memory.mark_visible(views, 'read')
    merged = memory.union_views(views)
    assert [(r['start'], r['end']) for r in merged] == [(0, 35), (50, 70)]
    assert memory.visible_characters == 55


def test_retained_feedback_uses_prior_raw_ranges_after_focus_change():
    index = Index()
    memory = EvidenceMemory(index.context, TokenClient(), 300)
    record = index.record(0)
    memory.observe({'results': [record]})
    view = memory.page(record, 0, record['end'], 300)
    memory.mark_visible([view], 'observation')
    result = reflection_context(memory, {'operator': 'graph_entities', 'results': [], 'status': 'ok'},
                                'Juniper location instead of acquisition', 300, 'reflection')
    assert [(r['id'], r['start'], r['end']) for r in result['existing_sources']] == [('C0', view['start'], view['end'])]
    assert result['available_sources'] == ['C0']


class RoutedIndex:
    """A sparse match can be lost by approximate centroid routing."""
    def __init__(self): self.lexical_calls = []
    def record(self, i, score):
        return {'id': f'C{i}', 'start': i*100, 'end': i*100+20, 'text': f'Independent source {i}', 'score': score}
    def execute(self, name, p):
        return {'operator': name, 'results': [self.record(i, 12-i) for i in [0, 1, 8, 9]]}
    def bm25(self, query, top_k, scope):
        self.lexical_calls.append((query, top_k, scope))
        return [(i, 12-i) for i in range(top_k)] if query != 'no matches' else []


def test_complete_lexical_list_competes_with_routed_candidates():
    api = KnowledgeAPI(RoutedIndex())
    result = api.execute('tree_search', {'query': 'archive custody', 'top_k': 4, 'scope': 'T7', 'mode': 'hybrid'})
    candidate = result['candidate_generation']
    assert candidate['lexical_ids'] == ['C0', 'C1', 'C2', 'C3']
    assert candidate['candidate_count'] == 6
    assert 'C2' in {r['id'] for r in result['results']}
    assert api.index.lexical_calls == [('archive custody', 4, 'T7')]


def test_hybrid_fusion_preserves_both_disjoint_candidate_lists():
    class Disjoint(RoutedIndex):
        def execute(self, name, p):
            return {'operator': name, 'results': [self.record(i, 12-i) for i in range(4, 8)]}
    result = KnowledgeAPI(Disjoint()).execute('tree_search', {'query': 'archive', 'top_k': 4, 'mode': 'hybrid'})
    ids = {r['id'] for r in result['results']}
    assert ids & {'C0', 'C1', 'C2', 'C3'} and ids & {'C4', 'C5', 'C6', 'C7'}


def test_tree_only_mode_preserves_route_and_scoped_cache_is_complete():
    api = KnowledgeAPI(RoutedIndex())
    p = {'query': 'archive', 'top_k': 4, 'scope': 'T7', 'mode': 'tree'}
    result = api.execute('tree_search', p)
    assert [r['id'] for r in result['results']] == ['C0', 'C1', 'C8', 'C9']
    assert not api.index.lexical_calls and 'candidate_generation' not in result
    api.execute('tree_search', {**p, 'mode': 'hybrid'})
    api.execute('tree_search', {**p, 'mode': 'hybrid'})
    assert len(api.index.lexical_calls) == 1


def test_pair_reranking_recalls_before_small_return_limit_discards_candidates():
    class Cross:
        def __init__(self): self.calls = []
        def scores(self, query, texts):
            self.calls.append((query, texts))
            return [10.0 if text.endswith('6') else -1.0 for text in texts]
    ranker = Cross()
    index = RoutedIndex()
    api = KnowledgeAPI(index, reranker=ranker, candidate_top_k=8)
    result = api.execute('tree_search', {'query': 'archive custody', 'top_k': 1, 'scope': 'T7', 'mode': 'hybrid'})
    assert len(result['results']) == 1 and result['results'][0]['id'] == 'C6'
    assert result['results'][0]['text'] == index.record(6, 1)['text']
    assert index.lexical_calls == [('archive custody', 8, 'T7')]
    reranking = result['candidate_generation']['pair_reranking']
    assert reranking['requested_result_limit'] == 1 and reranking['candidate_channel_limit'] == 8
    assert 'C6' in reranking['candidate_ids'] and len(reranking['candidate_ids']) > 1
    assert result['results'][0]['relevance_logit'] == 10.0 and result['results'][0]['score'] > 0
    api.execute('tree_search', {'query': 'archive custody', 'top_k': 1, 'scope': 'T7', 'mode': 'hybrid'})
    assert len(ranker.calls) == 1


def test_reranker_does_not_change_tree_only_route_or_pad_empty_recall():
    class Cross:
        def scores(self, query, texts):
            assert not texts
            return []
    api = KnowledgeAPI(RoutedIndex(), reranker=Cross(), candidate_top_k=8)
    result = api.execute('tree_search', {'query': 'archive', 'top_k': 1, 'mode': 'tree'})
    assert [r['id'] for r in result['results']] == ['C0', 'C1', 'C8', 'C9']
    class Empty(RoutedIndex):
        def execute(self, name, p): return {'operator': name, 'results': []}
    empty = KnowledgeAPI(Empty(), reranker=Cross(), candidate_top_k=8).execute('tree_search',
        {'query': 'no matches', 'top_k': 1, 'mode': 'hybrid'})
    assert empty['results'] == [] and empty['status'] == 'no_match'


def test_hybrid_does_not_pad_zero_match_lists():
    class Empty(RoutedIndex):
        def execute(self, name, p): return {'operator': name, 'results': [self.record(0, 0)]}
    result = KnowledgeAPI(Empty()).execute('tree_search', {'query': 'no matches', 'top_k': 4, 'mode': 'hybrid'})
    assert result['results'] == [] and result['status'] == 'no_match'


def test_context_restoration_preserves_anchors_and_stays_in_original_paragraph():
    prefix = 'Unrelated catalog procedure.\n\n'
    anchor = 'Birch Library keeps the register. '
    condition = 'Access requires an appointment. '
    text = prefix+anchor+condition+'\n\nOther records have different restrictions.'
    memory = EvidenceMemory(text, TokenClient(), 200)
    record = {'id': 'C0', 'start': 0, 'end': len(text), 'text': text}
    memory.observe({'results': [record]})
    view = memory.page(record, len(prefix), len(prefix)+len(anchor), 200)
    memory.mark_visible([view], 'read')
    _, final = memory.render('Birch register access', 200)
    assert len(final) == 1 and anchor in final[0]['text'] and condition in final[0]['text']
    assert final[0]['start'] == len(prefix)
    assert final[0]['end'] <= text.index('Other records')
    assert memory.selections[-1]['anchor_ranges'] == [{'id': 'C0', 'start': view['start'], 'end': view['end']}]
    assert memory.selections[-1]['tokens'] <= 200


def test_context_restoration_never_sacrifices_an_anchor_to_fit_more_text():
    text = 'Birch Library keeps the register. Access requires an appointment. '
    memory = EvidenceMemory(text, TokenClient())
    record = {'id': 'C0', 'start': 0, 'end': len(text), 'text': text}
    memory.observe({'results': [record]})
    end = text.index('Access')
    view = memory.page(record, 0, end, 200)
    minimum = memory.count(memory.block(view))
    for budget in [minimum, minimum+4, 200]:
        result = memory.restore_context([view], budget)
        assert result[0]['start'] <= view['start'] and result[0]['end'] >= view['end']
        assert memory.count('\n\n'.join(memory.block(v) for v in result)) <= budget
        assert all(text[v['start']:v['end']] == v['text'] for v in result)


def test_context_restoration_respects_embedded_source_bounds():
    text = 'Earlier source.\n\nAster register has a custodian. Inspection requires permission.\n\nLater source.'
    start, end = text.index('Aster'), text.index('Later')
    record = {'id': 'C1', 'start': start, 'end': end, 'text': text[start:end]}
    memory = EvidenceMemory(text, TokenClient())
    memory.observe({'results': [record]})
    anchor = memory.page(record, start, text.index('Inspection'), 200)
    restored = memory.restore_context([anchor], 300)
    assert all(start <= r['start'] < r['end'] <= end for r in restored)
    assert 'Earlier' not in restored[0]['text'] and 'Later' not in restored[0]['text']


def test_source_entity_descriptors_rank_all_returned_ids_before_clipping():
    class Graph:
        entities = list(range(31))
        postings = {}
        def describe(self, eid):
            n = int(eid[1:]); name = 'Linden inventory' if n == 30 else f'Catalog section{n}'
            return {'id': eid, 'name': name, 'canonical': name.lower(), 'aliases': [name],
                    'passage_ids': ['C0', 'C1'], 'source_count': 2}
    class EntityIndex(Index):
        entity_graph = Graph()
        def record(self, i):
            r = super().record(i)
            r['entity_ids'] = [f'E{n}' for n in range(31)]
            r['entities'] = [self.entity_graph.describe(eid) for eid in r['entity_ids'][:24]]
            return r
    api = KnowledgeAPI(EntityIndex())
    result = api.execute('tree_search', {'query': 'Linden inventory', 'top_k': 8})
    record = result['results'][0]
    assert record['entities'][0]['id'] == 'E30'
    assert len(record['entities']) == 24
    assert all(e['id'] in record['entity_ids'] for e in record['entities'])
    selection = record['entity_descriptor_selection']
    assert selection['query'] == 'Linden inventory' and len(selection['candidate_ids']) == 31
    other = api.execute('tree_search', {'query': 'Catalog section0', 'top_k': 8})
    assert other['results'][0]['entities'][0]['id'] != 'E30'


def test_optional_semantic_channel_can_recall_disjoint_source_without_forcing_route():
    class Dense:
        def __init__(self): self.calls = []
        def search(self, query, limit, scope):
            self.calls.append((query, limit, scope))
            return [RoutedIndex().record(i, .9-i/100) for i in range(4, 8)]
    dense = Dense()
    api = KnowledgeAPI(RoutedIndex(), dense)
    p = {'query': 'archive custody', 'scope': 'T7', 'top_k': 4, 'mode': 'hybrid'}
    result = api.execute('tree_search', p)
    ids = {r['id'] for r in result['results']}
    assert ids & {'C0', 'C1', 'C2', 'C3'} and ids & {'C4', 'C5', 'C6', 'C7'}
    assert dense.calls == [('archive custody', 4, 'T7')]
    api.execute('tree_search', p)
    assert len(dense.calls) == 1
    api.execute('tree_search', {**p, 'mode': 'tree'})
    assert len(dense.calls) == 1
    assert result['candidate_generation']['semantic_channel']['selected_ids'] == [r['id'] for r in result['results']]


def test_protocol_demonstrations_never_enter_observations_or_evidence_memory():
    class Inspect(Client):
        def __init__(self):
            super().__init__([select('tree_search')], stop_after=1)
            self.reflection_messages = None
        def complete(self, messages, maximum, **kwargs):
            if 'reflection' in kwargs['tag']:
                self.reflection_messages = deepcopy(messages)
            return super().complete(messages, maximum, **kwargs)
    client = Inspect()
    result = research('Trace the Willow collection acquisition', Index(), client,
                      {'reflection_protocol_examples': True})
    messages = client.reflection_messages
    assert len(messages) == 6
    assert json.loads(messages[1]['content'])['protocol_demonstration']
    assert json.loads(messages[3]['content'])['protocol_demonstration']
    actual = json.loads(messages[-1]['content'])
    assert actual['available_sources'] == ['C0'] and 'protocol_demonstration' not in actual
    assert {r['id'] for r in result.evidence_records} == {'C0'}
    assert all(span['id'] == 'C0' for e in result.state['exposures'] for span in e['spans'])
    assert len(result.trace[0]['executed']) == 1
    assert result.state['reflection_protocol_examples']['demonstrations_are_task_evidence'] is False


def test_disabled_protocol_examples_preserve_original_message_shape():
    from repar.v4.protocol import reflection_messages
    from repar.workflow import REFLECTION_PROMPT
    payload = {'task': 'Inspect sources', 'observation': {'source_views': []}, 'available_sources': []}
    before = deepcopy(payload)
    messages = reflection_messages(payload, False)
    assert messages == [{'role': 'system', 'content': REFLECTION_PROMPT},
                        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    assert payload == before


def test_search_budget_options_are_available_without_task_classification():
    memory = EvidenceMemory(Index.context, TokenClient())
    config = {'retrieval_budget_options': [1, 8]}
    for task in ['Trace the Willow history', 'A free-form research request']:
        actions = action_candidates(task, {'query': 'Willow acquisition'}, memory, set(), config, 0)
        searches = [a for a in actions if a['name'] == 'tree_search']
        assert {(a['parameters']['top_k'], bool(a['parameters'].get('anchor_query'))) for a in searches} == {
            (1, False), (8, False), (1, True), (8, True)}
        assert {a['name'] for a in actions} >= {'tree_search', 'graph_entities', 'tree_browse'}


def test_model_selected_search_limit_is_executed_exactly():
    index = Index()
    client = Client([select('tree_search', top_k=1)], stop_after=1)
    result = research('Trace the acquisition', index, client, {'retrieval_budget_options': [1, 8]})
    decision = result.trace[0]['decision']
    assert decision['parameters']['top_k'] == 1
    assert result.trace[0]['executed'][0]['observation']['executed_parameters'] == decision['parameters']
    assert any(name == 'tree_search' and p['top_k'] == 1 for name, p in index.calls)


def test_factored_resource_decision_precedes_real_observation_and_reflection():
    class ParameterClient(Client):
        def complete(self, messages, maximum, **kwargs):
            if 'parameters' in kwargs['tag']:
                self.events.append((kwargs['tag'], json.loads(messages[-1]['content']), deepcopy(kwargs)))
                return json.dumps({'top_k': 1})
            return super().complete(messages, maximum, **kwargs)
    index = Index()
    client = ParameterClient([select('tree_search', scope='T0')], stop_after=1)
    result = research('Trace the acquisition', index, client,
        {'retrieval_budget_options': [1, 8], 'factor_action_parameters': True})
    assert [e[0] for e in client.events] == ['v4_plan_1', 'v4_action_1', 'v4_parameters_1', 'v4_reflection_1']
    assert result.trace[0]['resource_selection_model_call']
    executed = result.trace[0]['executed'][0]
    assert executed['parameters']['top_k'] == 1 and executed['observation']['executed_parameters'] == executed['parameters']
    assert client.events[-1][1]['observation']['source_views']


def test_completed_budget_call_does_not_remove_other_legal_limits():
    memory = EvidenceMemory(Index.context, TokenClient())
    cfg = {'retrieval_budget_options': [1, 8]}
    before = action_candidates('Find archive history', {'query': 'Willow'}, memory, set(), cfg, 0)
    one = next(a for a in before if a['name'] == 'tree_search' and a['parameters']['top_k'] == 1)
    after = action_candidates('Find archive history', {'query': 'Willow'}, memory,
        {request_identity(one['name'], one['parameters'])}, cfg, 1)
    assert not any(a['name'] == one['name'] and a['parameters'] == one['parameters'] for a in after)
    assert any(a['name'] == 'tree_search' and a['parameters']['top_k'] == 8 for a in after)


@pytest.mark.parametrize('options', [[], [0], [25], [True], [1.5], '1,8'])
def test_invalid_search_budget_config_fails_before_any_model_call(options):
    client = Client([])
    with pytest.raises(ValueError, match='Retrieval budget options'):
        research('Find archive sources', Index(), client, {'retrieval_budget_options': options})
    assert client.events == []
