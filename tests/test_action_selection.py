"""General action-family binding, including repeat and observation boundaries."""
from copy import deepcopy
import json

from repar.v4.selection import action_families, choose_action


def actions():
    return [
        {'action_id': 'a0', 'name': 'tree_search', 'parameters': {'query': 'Juniper ledger', 'scope': 'T0', 'top_k': 1}, 'purpose': 'Narrow retrieval'},
        {'action_id': 'a1', 'name': 'tree_search', 'parameters': {'query': 'Juniper ledger', 'scope': 'T0', 'top_k': 8}, 'purpose': 'Broad retrieval'},
        {'action_id': 'a2', 'name': 'tree_search', 'parameters': {'query': 'Juniper ledger', 'scope': 'T3', 'top_k': 1}, 'purpose': 'Other scope'},
        {'action_id': 'a3', 'name': 'read', 'parameters': {'ids': ['C7'], 'radius': 1}, 'purpose': 'Read source'},
        {'action_id': 'answer', 'name': 'answer', 'parameters': {}, 'purpose': 'Answer'},
    ]


class Client:
    def __init__(self, family='g0', limit=8):
        self.family, self.limit, self.calls = family, limit, []
    def complete(self, messages, maximum, **kwargs):
        self.calls.append({'messages': messages, **kwargs})
        return json.dumps({'top_k': self.limit} if 'parameters' in kwargs['tag'] else {'action_id': self.family})


def test_families_preserve_exact_variants_and_reference_boundaries():
    original = actions(); before = deepcopy(original)
    families = action_families(original)
    assert len(families) == 4
    assert [v['action_id'] for v in families[0]['variants']] == ['a0', 'a1']
    assert families[0]['parameters'] == {'query': 'Juniper ledger', 'scope': 'T0'}
    assert families[1]['parameters']['scope'] == 'T3'
    assert [f['name'] for f in families] == ['tree_search', 'tree_search', 'read', 'answer']
    assert original == before
    families[0]['variants'][0]['parameters']['query'] = 'mutated'
    assert original == before


def test_model_chosen_cap_materializes_an_exact_available_call():
    client = Client(limit=8)
    call, metadata = choose_action(client, 'Trace Juniper records', {'objective': 'Collect custody statements', 'query': 'Juniper ledger'},
        None, actions(), {'factor_action_parameters': True}, 0)
    assert call == actions()[1]
    assert metadata['selected_family'] == 'g0' and metadata['resource_selection_model_call']
    assert len(client.calls) == 2
    payload = json.loads(client.calls[1]['messages'][-1]['content'])
    assert set(payload) == {'task', 'objective', 'result_limits'}
    assert client.calls[1]['schema']['properties']['top_k']['enum'] == [1, 8]
    assert all('observation' not in json.loads(c['messages'][-1]['content']) for c in client.calls)


def test_completed_variant_cannot_be_resurrected_by_parameter_binding():
    available = [a for a in actions() if a['action_id'] != 'a1']
    client = Client(limit=8)
    call, metadata = choose_action(client, 'Inspect the ledger', {'objective': 'Locate its source', 'query': 'Juniper'},
        None, available, {'factor_action_parameters': True}, 2)
    assert call['parameters']['top_k'] == 1
    assert len(client.calls) == 1 and not metadata['resource_selection_model_call']


def test_read_and_answer_require_no_resource_subdecision():
    for family, expected in [('g2', 'read'), ('g3', 'answer')]:
        client = Client(family)
        call, metadata = choose_action(client, 'Trace custody', {'objective': 'Inspect next source', 'query': 'Juniper'},
            None, actions(), {'factor_action_parameters': True}, 0)
        assert call['name'] == expected and len(client.calls) == 1
        assert not metadata['resource_selection_model_call']


def test_legacy_direct_action_choice_keeps_original_shape():
    client = Client('a1')
    call, metadata = choose_action(client, 'Trace custody', {'objective': 'Inspect source', 'query': 'Juniper'},
        None, actions(), {}, 0)
    assert call == actions()[1] and metadata == {} and len(client.calls) == 1
    payload = json.loads(client.calls[0]['messages'][-1]['content'])
    assert payload['available_actions'] == actions()
