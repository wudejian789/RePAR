"""Declared generation settings must enter real requests and cache identities."""
from copy import deepcopy

import pytest

from repar.client import LLMClient, sampling_parameters


class Client(LLMClient):
    def __init__(self, cfg, cache):
        super().__init__(cfg, cache); self.requests = []
    def post(self, route, body, **kwargs):
        self.requests.append(deepcopy(body))
        return {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}


def config(**extra):
    return {'base_url': 'http://independent-fixture', 'model': 'fixture', 'temperature': .1,
            'seed': 42, 'enable_thinking': False, 'deployment_id': 'fixture-r1', **extra}


def test_control_sampler_does_not_change_unstructured_answer(tmp_path):
    profile = {'temperature': 1.0, 'top_k': 20, 'top_p': 1.0, 'min_p': 0.0, 'presence_penalty': 2.0, 'repeat_penalty': 1.0}
    client = Client(config(control_sampling=profile), tmp_path)
    client.complete([{'role': 'user', 'content': 'Inspect the record.'}], 128, json_mode=True)
    client.complete([{'role': 'user', 'content': 'Explain the record.'}], 256)
    assert all(client.requests[0][k] == v for k, v in profile.items())
    assert client.requests[1]['temperature'] == .1 and 'top_k' not in client.requests[1]
    assert client.requests[1]['seed'] == 42 and not client.requests[1]['chat_template_kwargs']['enable_thinking']


def test_actual_sampler_payload_changes_cache_identity_and_reuses_only_exact_request(tmp_path):
    message = [{'role': 'user', 'content': 'Describe archive custody.'}]
    old = Client(config(), tmp_path); old.complete(message)
    new = Client(config(sampling={'top_k': 20}), tmp_path); new.complete(message)
    assert new.calls[0]['key'] != old.calls[0]['key'] and not new.calls[0]['cached']
    again = Client(config(sampling={'top_k': 20}), tmp_path); again.complete(message)
    assert again.calls[0]['cached'] and not again.requests


def test_omitted_sampler_fields_preserve_legacy_request(tmp_path):
    client = Client(config(), tmp_path)
    client.complete([{'role': 'user', 'content': 'Inspect sources.'}], 128, json_mode=True)
    assert client.requests[0] == {'model': 'fixture', 'messages': [{'role': 'user', 'content': 'Inspect sources.'}],
        'temperature': .1, 'seed': 42, 'max_tokens': 128, 'stream': False,
        'chat_template_kwargs': {'enable_thinking': False}, 'response_format': {'type': 'json_object'}}


@pytest.mark.parametrize('supplied', [{'model': 'another'}, {'top_k': 1.5}, {'min_p': 2}, {'top_p': -1},
    {'temperature': -1}, {'repeat_penalty': 0}, {'presence_penalty': 3}, {'temperature': 'one'},
    {'temperature': float('nan')}, {'repeat_penalty': float('inf')}])
def test_invalid_or_identity_overriding_sampler_is_rejected(supplied):
    with pytest.raises(ValueError):
        sampling_parameters(config(control_sampling=supplied), structured=True)
