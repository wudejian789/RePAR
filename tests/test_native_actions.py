"""Native tool protocol identity, bound references and pre-execution repair."""
from copy import deepcopy
import json

import pytest

from repar.client import LLMClient
from repar.v4.native import declarations, select_native, parse_native, native_payload
from repar.v4.selection import choose_action


def menu():
    return [
        {'name': 'tree_search', 'action_id': 'g0', 'parameters': {'query': 'Larch custody'}, 'purpose': 'Search sources'},
        {'name': 'read', 'action_id': 'g1', 'parameters': {'ids': ['C8'], 'radius': 0}, 'purpose': 'Read observed source'},
    ]


def message(name, action=None):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't1', 'type': 'function',
        'function': {'name': name, 'arguments': json.dumps({'binding': action} if action is not None else {})}}]}


class Client:
    def __init__(self, responses): self.responses, self.calls = iter(responses), []
    def complete_tool(self, messages, tools, maximum, **kwargs):
        self.calls.append((deepcopy(messages), deepcopy(tools), maximum, kwargs))
        return next(self.responses)


def test_native_declarations_keep_action_ids_with_their_real_operator():
    tools = declarations(menu())
    assert [t['function']['name'] for t in tools] == ['tree_search', 'read']
    assert all(t['function']['parameters']['properties'] == {} for t in tools)


def test_local_bindings_do_not_depend_on_other_operator_positions():
    actions = menu()+[{'name': 'read', 'action_id': 'g7', 'parameters': {'ids': ['C9']}, 'purpose': 'Other source'}]
    choice, metadata = parse_native(message('read', 'b1'), actions)
    assert choice == {'action_id': 'g7'} and metadata['native_binding'] == 'b1'
    reversed_menu = [actions[1], actions[0], actions[2]]
    assert parse_native(message('read', 'b1'), reversed_menu) == (choice, metadata)
    payload = native_payload('Read a source', {'query': 'Larch'}, None, actions)
    assert payload['available_bindings']['read'][1]['parameters']['ids'] == ['C9']
    assert 'g7' not in json.dumps(payload) and 'g1' not in json.dumps(payload)


def test_native_selection_is_a_real_bound_call_not_an_ordinary_answer():
    client = Client([message('read')])
    result, state = select_native(client, 'Inspect Larch records', {'query': 'Larch custody'}, None, menu(), 128, 'v4_action_1')
    assert result == {'action_id': 'g1'} and state == {'native_function': 'read', 'native_binding': 'b0'}
    assert len(client.calls) == 1 and client.calls[0][3]['tag'] == 'v4_action_1'


@pytest.mark.parametrize('invalid', [message('read', 'g0'), message('invented_tool', 'g0'),
    {'role': 'assistant', 'content': 'An unsupported final answer'},
    {'role': 'assistant', 'tool_calls': [message('read')['tool_calls'][0]]*2}])
def test_invalid_control_repaired_before_any_tool_or_observation(invalid):
    client = Client([invalid, message('read')])
    choice, _ = select_native(client, 'Inspect sources', {'query': 'Larch'}, None, menu(), 128, 'v4_action_1')
    assert choice == {'action_id': 'g1'} and len(client.calls) == 2
    assert client.calls[-1][3]['tag'] == 'v4_action_1_repair'
    assert client.calls[-1][0][-1]['role'] == 'user'
    assert 'No tool has executed' in client.calls[-1][0][-1]['content']
    assert not any(m['role'] == 'tool' for call in client.calls for m in call[0])


def test_repeated_invalid_native_control_raises_without_answer_recovery():
    client = Client([{'content': 'answer'}, {'content': 'answer'}])
    with pytest.raises(ValueError, match='Invalid native action'):
        select_native(client, 'Inspect sources', {'query': 'Larch'}, None, menu(), 128, 'v4_action_1')


def test_native_client_cache_has_tools_and_deployment_identity(tmp_path):
    class HTTP(LLMClient):
        def post(self, route, body, **kwargs):
            self.body = deepcopy(body)
            self.count = getattr(self, 'count', 0)+1
            return {'choices': [{'message': message('read')}]}
    config = {'base_url': 'http://localhost:9091', 'model': 'example', 'deployment_id': 'one',
        'seed': 42, 'temperature': .1, 'enable_thinking': False, 'answer_grammar': 'unused'}
    client = HTTP(config, tmp_path)
    messages = [{'role': 'user', 'content': 'Inspect original references'}]
    result = client.complete_tool(messages, declarations(menu()), 128, tag='native')
    client.complete_tool(messages, declarations(menu()), 128, tag='native_again')
    assert result['tool_calls'][0]['function']['name'] == 'read' and client.count == 1
    assert client.calls[1]['cached'] and client.body['tool_choice'] == 'required'
    assert client.body['parallel_tool_calls'] is False and 'grammar' not in client.body
    other = HTTP({**config, 'deployment_id': 'two'}, tmp_path)
    other.complete_tool(messages, declarations(menu()), 128)
    assert other.count == 1 and other.calls[0]['key'] != client.calls[0]['key']


def test_native_family_selection_materializes_original_concrete_read():
    actions = [{'name': 'tree_search', 'action_id': 'a0', 'parameters': {'query': 'Larch', 'top_k': 8}, 'purpose': 'Search'},
               {'name': 'read', 'action_id': 'a1', 'parameters': {'ids': ['C8'], 'radius': 0}, 'purpose': 'Read original'}]
    client = Client([message('read')])
    selected, state = choose_action(client, 'Read the located source', {'objective': 'Inspect source', 'query': 'Larch'},
        None, actions, {'factor_action_parameters': True, 'native_action_tools': True}, 0)
    assert selected == actions[1] and state['native_function'] == 'read'
    assert not state['resource_selection_model_call']
