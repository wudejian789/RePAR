"""Resolve observed entity provenance to real source text without forcing a route."""
from repar.v4.memory import EvidenceMemory
from repar.v4.workflow import action_candidates
from repar.v4.knowledge import request_identity
from repar.v4.workflow import unread_source_references


class Tokens:
    def tokenize(self, text): return text.split()


def test_entity_origin_sources_are_readable_before_they_have_been_loaded():
    memory = EvidenceMemory('Original knowledge source', Tokens())
    observation = {'operator': 'graph_entities', 'status': 'ok', 'results': [], 'entities': [
        {'id': 'E4', 'name': 'Willow register', 'passage_ids': ['C2', 'C9'], 'source_count': 2}]}
    memory.observe(observation)
    actions = action_candidates('Trace Willow custody', {'query': 'Willow register'}, memory, set(),
        {'read_entity_sources': True}, 0)
    read = next(a for a in actions if a['name'] == 'read')
    assert read['parameters'] == {'ids': ['C2', 'C9'], 'radius': 0} and read['provenance_entity'] == 'E4'
    assert {a['name'] for a in actions} >= {'tree_search', 'graph_entities', 'graph_neighbors', 'read', 'tree_browse'}
    assert not memory.records and memory.visible_characters == 0 and not memory.source_views


def test_origin_calls_are_bounded_deduplicated_and_excluded_after_execution():
    memory = EvidenceMemory('', Tokens())
    memory.entities = {eid: {'id': eid, 'name': 'Archive', 'passage_ids': ['C7', 'C7', 'C9', 'C11']}
                       for eid in ['E1', 'E2']}
    config = {'read_entity_sources': True, 'retrieval_top_k': 2}
    before = action_candidates('Find archival source', {'query': 'Archive'}, memory, set(), config, 0)
    reads = [a for a in before if a['name'] == 'read']
    assert len(reads) == 1 and reads[0]['parameters']['ids'] == ['C7', 'C9']
    identity = request_identity('read', reads[0]['parameters'])
    after = action_candidates('Find archival source', {'query': 'Archive'}, memory, {identity}, config, 1)
    assert not any(a['name'] == 'read' for a in after)


def test_entity_reference_visibility_does_not_claim_that_source_text_is_visible():
    entity = {'id': 'E1', 'name': 'Archive', 'passage_ids': ['C4', 'C5'], 'source_count': 2}
    memory = EvidenceMemory('', Tokens(), entity_source_references=True)
    observation = {'operator': 'graph_entities', 'status': 'ok', 'entities': [entity], 'results': []}
    memory.observe(observation)
    view = memory.observation_view(observation, 'Archive', 100, 'entities')
    assert view['entities'][0]['passage_ids'] == ['C4', 'C5']
    assert view['source_views'] == [] and not memory.records and not memory.source_views
    old = EvidenceMemory('', Tokens())
    assert 'passage_ids' not in old.observation_view(observation, 'Archive', 100, 'entities')['entities'][0]


def test_source_references_are_unique_unread_navigation_handles():
    memory = EvidenceMemory('record', Tokens())
    memory.observe({'entities': [{'id': 'E1', 'passage_ids': ['C1', 'C2', 'C1']}]})
    assert unread_source_references(memory) == {'C1', 'C2'}
    memory.observe({'entities': [{'id': 'E3', 'passage_ids': ['C2']}], 'results': [
        {'id': 'C1', 'start': 0, 'end': 6, 'text': 'record'}]})
    assert unread_source_references(memory) == {'C2'}
    assert memory.visible_characters == 0


def test_relational_navigation_remains_available_after_reading_seed_origins():
    memory = EvidenceMemory('The registry was transferred.', Tokens())
    memory.observe({'results': [{'id': 'C0', 'start': 0, 'end': len(memory.context), 'text': memory.context,
                                'entities': [{'id': 'E1', 'name': 'Registry', 'passage_ids': ['C0'], 'relation_count': 1},
                                             {'id': 'E2', 'name': 'Copy', 'passage_ids': ['C0'], 'relation_count': 0}]}]})
    actions = action_candidates('Trace the registry transfer', {'query': 'Registry'}, memory, set(),
                                {'agent_version': 'react-v4-r20', 'read_entity_sources': True}, 0)
    traversal = [action for action in actions if action['name'] == 'graph_neighbors']
    assert len(traversal) == 1
    assert traversal[0]['parameters']['seeds'] == ['E1']
    assert traversal[0]['parameters']['exclude_passages'] == ['C0']
    assert not any('provenance_entity' in action for action in actions)
