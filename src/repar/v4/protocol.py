"""Abstract response-protocol demonstrations, never task evidence or memory."""
import json

from repar.workflow import REFLECTION_PROMPT

DEMONSTRATIONS = [
    {'task': 'Report the first and second attributes of record R.',
     'plan': {'objective': 'Locate the requested attributes', 'query': 'record R attributes'},
     'action': {'name': 'read', 'parameters': {'ids': ['C17'], 'radius': 0}},
     'observation': {'operator': 'read', 'status': 'ok', 'source_views': [
         {'id': 'C17', 'text': 'Record R has first attribute: amber.'}]},
     'existing_sources': [], 'available_sources': ['C17'], 'protocol_demonstration': True},
    {'assessment': 'C17 supplies the first attribute only.', 'gap': 'The second attribute has not been provided.',
     'lesson': 'Look up the missing attribute without repeating the known one.',
     'next': 'continue', 'supported_sources': ['C17']},
    {'task': 'Report the first and second attributes of record R.',
     'plan': {'objective': 'Locate the missing second attribute', 'query': 'record R second attribute'},
     'action': {'name': 'read', 'parameters': {'ids': ['C18'], 'radius': 0}},
     'observation': {'operator': 'read', 'status': 'ok', 'source_views': [
         {'id': 'C18', 'text': 'Record R has second attribute: cedar.'}]},
     'existing_sources': [{'id': 'C17', 'text': 'Record R has first attribute: amber.'}],
     'available_sources': ['C17', 'C18'], 'protocol_demonstration': True},
    {'assessment': 'C17 and C18 provide both requested attributes.', 'gap': 'none',
     'lesson': 'The available source statements cover the request.', 'next': 'answer', 'supported_sources': ['C17', 'C18']},
]


def reflection_messages(payload, demonstrate=False, *, batch=False):
    system = REFLECTION_PROMPT
    if batch:
        system += '\nUse all observations from this round together. Report the specific evidence still missing when continuing.'
    messages = []
    if demonstrate:
        system += '\nDemonstration turns illustrate only the JSON protocol. Only the final user message contains current task evidence.'
        messages = [{'role': 'user' if i % 2 == 0 else 'assistant',
                     'content': json.dumps(value, ensure_ascii=False)} for i, value in enumerate(DEMONSTRATIONS)]
    return [{'role': 'system', 'content': system}, *messages,
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
