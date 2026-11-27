import json
from copy import deepcopy

import pytest

pytest.importorskip("spacy",reason="Install the graph extra to test v2 extraction")
pytest.importorskip("en_core_web_sm",reason="Install the pinned English extraction model to test v2")

from repar.index_v2 import KnowledgeIndex
from repar.workflow import research,action_candidates,signature


CONFIG={"chunk_words":40,"tree_fanout":4,"graph_neighbors":3,"index_version":"entity-graph-v2",
        "graph_model":"en_core_web_sm","react_max_steps":4,"evidence_word_budget":2000}


@pytest.fixture(scope="module")
def kb():
    context=("The Delta Commission developed the Silver Protocol to monitor river pollution. "
             "Independent observers reviewed technical standards for drinking water and filed the annual report.\n\n"
             "The museum catalog lists paintings and historic architecture. The annual exhibition includes bronze sculptures. "
             "Visitors can access the collection on weekdays.\n\n"
             "Harbor Council adopted the Silver Protocol in 2024. Harbor Council funds mobile clinics. "
             "Mobile clinics reduce travel costs for residents.\n\n"
             "North Agency did not adopt the Silver Protocol in 2023. North Agency may fund pilot programs.")
    return KnowledgeIndex(context,CONFIG)


def test_entity_graph_has_many_to_many_mapping_and_cross_source_predicates(kb):
    graph=kb.entity_graph
    silver=next(e for e in graph.entities if e["canonical"]=="silver protocol")
    assert len(silver["passage_ids"])>=2
    assert any(len(ids)>1 for ids in graph.passage_entities)
    paths=kb.graph_neighbors([silver["id"]],query="Which organization adopted it?",hops=1)
    triples={(graph.entity(r["subject"])["canonical"],r["predicate"],graph.entity(r["object"])["canonical"]) for r in paths["relations"]}
    assert ("delta commission","develop","silver protocol") in triples
    assert ("harbor council","adopt","silver protocol") in triples
    assert len({r["id"] for r in paths["results"]})>=2
    assert all(r["status"]=="automatically_extracted_not_verified" for r in paths["relations"])
    for p in graph.propositions:
        assert kb.context[p["start"]:p["end"]]==p["text"]
    for e in graph.entities:
        for mention in e["mentions"]:
            assert kb.context[mention["start"]:mention["end"]]==mention["text"]
    assert any(r["negated"] and r["predicate"]=="adopt" for r in graph.relations)
    assert any(r["modality"]==["may"] for r in graph.relations)
    assert any("2024" in r["time"] for r in graph.relations)


def test_invalid_graph_ids_and_direction_are_explicit(kb):
    with pytest.raises(ValueError,match="Unknown entity ID"):
        kb.graph_neighbors(["Paper 1"])
    with pytest.raises(ValueError,match="direction"):
        kb.graph_neighbors(["E0"],direction="anything")
    with pytest.raises(ValueError,match="disabled"):
        kb.execute("graph_entities",{"query":"water"},allow_graph=False)


def test_entity_direction_and_predicate_filter_preserve_negation(kb):
    graph=kb.entity_graph
    silver=next(e for e in graph.entities if e["canonical"]=="silver protocol")
    incoming=kb.graph_neighbors([silver["id"]],hops=1,direction="incoming",kinds=["predicate"])
    assert incoming["relations"]
    assert all(r["object"]==silver["id"] for r in incoming["relations"] if r["id"] in
               {step["relation_id"] for path in incoming["paths"] for step in path["steps"]})
    negative=[r for r in incoming["relations"] if r["negated"]]
    assert negative and all("2023" in r["time"] for r in negative)


@pytest.mark.parametrize('direction,kinds,wrong_direction,wrong_kind', [
    ('incoming', ['predicate'], 'outgoing', 'predicate'),
    ('both', ['co_mention'], 'incoming', 'predicate'),
])
def test_graph_filters_only_eligible_edges_before_the_budget(kb, direction, kinds, wrong_direction, wrong_kind):
    from repar.v4.knowledge import KnowledgeAPI

    index = deepcopy(kb)
    graph = index.entity_graph
    seed = next(e['id'] for e in graph.entities if e['canonical'] == 'silver protocol')
    other = next(e['id'] for e in graph.entities if e['canonical'] == 'harbor council')
    original = next(r for r in graph.relations
                    if r['subject'] == other and r['object'] == seed and r['predicate'] == 'adopt')
    expected = original['passage_ids'][0]
    eligible_id = f'R{len(graph.relations) + 32}'
    graph.adjacency[seed] = []
    graph.entity(seed)['proposition_ids'] = []
    for _ in range(32):
        rid = f'R{len(graph.relations)}'
        graph.relations.append({**original, 'id': rid, 'kind': wrong_kind})
        graph.adjacency[seed].append((rid, other, wrong_direction))
    graph.relations.append({**original, 'id': eligible_id, 'kind': kinds[0]})
    graph.adjacency[seed].append((eligible_id, other, 'incoming'))
    params = {'seeds': [seed], 'query': 'Harbor Council adopted the Silver Protocol',
              'hops': 1, 'top_k': 3, 'kinds': kinds, 'direction': direction}
    old = KnowledgeAPI(index, r3_retrieval=True).execute('graph_neighbors', params)
    new = KnowledgeAPI(index, r3_retrieval=True, eligible_graph_edges_first=True).execute('graph_neighbors', params)
    assert old['results'] == []
    assert expected in {record['id'] for record in new['results']}
    assert [step['relation_id'] for path in new['paths'] for step in path['steps']] == [eligible_id]
    assert all(index.context[r['start']:r['end']] == r['text'] for r in new['results'])


class FakePhases:
    def __init__(self):self.calls=[];self.graph_chosen=False
    def complete(self,messages,max_tokens,json_mode=False,tag="",schema=None):
        body=json.loads(messages[-1]["content"])
        self.calls.append((tag,deepcopy(body)))
        if "plan" in tag:
            return json.dumps({"objective":"Trace the protocol from its developer to its adopter","query":"Silver Protocol"})
        if "action" in tag:
            actions=body["available_actions"]
            names=[a["name"] for a in actions]
            desired="tree_search" if not any("reflection" in t for t,_ in self.calls) else "graph_neighbors"
            chosen=next((a for a in actions if a["name"]==desired),next(a for a in actions if a["name"]!="answer"))
            self.graph_chosen=chosen["name"]=="graph_neighbors"
            return json.dumps({"action_id":chosen["action_id"]})
        assert "reflection" in tag
        assert "observation" in body and "available_sources" in body
        return json.dumps({"assessment":"The returned source supports this step","gap":"Find another connected source",
                           "lesson":"Follow the shared entity","next":"answer" if self.graph_chosen else "continue",
                           "supported_sources":body["available_sources"][:2]})


def test_phases_run_in_true_order_and_graph_calls_bind_existing_entities(kb):
    client=FakePhases()
    result=research("Who adopted the protocol developed by the Delta Commission?",kb,client,CONFIG)
    assert [tag.split('_')[1] for tag,_ in client.calls]==['plan','action','reflection','plan','action','reflection']
    assert result.termination=="model_sufficient"
    assert result.trace[0]["executed"][0]["name"]=="tree_search"
    assert result.trace[1]["executed"][0]["name"]=="graph_neighbors"
    assert not any(c.get("error") for s in result.trace for c in s["executed"])
    assert result.evidence_ids
    # Second plan sees the actual post-execution assessment and its lesson.
    assert client.calls[3][1]["last_reflection"]["lesson"]=="Follow the shared entity"


def test_completed_actions_removed_before_selection_and_no_graph_ablation(kb):
    from repar.memory import EvidenceMemory
    memory=EvidenceMemory(1000);memory.entities={}
    first=action_candidates(kb,{"query":"Silver Protocol"},memory,set(),True,0)
    tree=next(a for a in first if a["name"]=="tree_search")
    second=action_candidates(kb,{"query":"Protocol Silver"},memory,{signature(tree)},True,1)
    assert not any(a["name"]=="tree_search" for a in second)
    ablated=action_candidates(kb,{"query":"Silver Protocol"},memory,set(),False,0)
    assert not any(a["name"].startswith("graph_") for a in ablated)


def test_one_source_can_stop_without_forcing_another_retrieval(kb):
    class Single(FakePhases):
        def complete(self,*args,**kwargs):
            text=super().complete(*args,**kwargs)
            if "reflection" in kwargs.get("tag",""):
                value=json.loads(text);value["next"]="answer";return json.dumps(value)
            return text
    result=research("Summarize the Silver Protocol source",kb,Single(),CONFIG)
    assert len(result.trace)==1 and result.termination=="model_sufficient"
