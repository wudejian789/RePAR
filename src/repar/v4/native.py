"""Select existing bound knowledge calls using the model's native tool protocol."""
import json


NATIVE_PROMPT = ('You are selecting a knowledge API operation. Make exactly one of the available function calls '
    'to address the current plan using only the provided bound references. Do not answer the task in ordinary prose; '
    'actual source text will be returned after your tool executes. Source references are locations, not source facts. '
    'Previous model assessments are unverified. Select answer only when the current evidence supports the task.')

CATALOG = {
    'tree_search': 'Search original source passages relevant to a query.',
    'tree_browse': 'Inspect known source-tree nodes and their child sections.',
    'read': 'Read original text at source passage references that were already located.',
    'graph_entities': 'Resolve names or concepts to source-linked entity references.',
    'graph_neighbors': 'Follow documented relations from observed entity references to other entities and sources.',
    'answer': 'Stop information gathering and answer using the current source evidence.',
}


def declarations(menu):
    names = list(dict.fromkeys(item['name'] for item in menu))
    tools = []
    for name in names:
        bound = [item for item in menu if item['name'] == name]
        properties = {'binding': {'type': 'string', 'enum': [f'b{i}' for i in range(len(bound))]}} if len(bound) > 1 else {}
        tools.append({'type': 'function', 'function': {'name': name, 'description': CATALOG.get(name, name),
            'parameters': {'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}, 'strict': True}})
    return tools


def native_payload(task, plan, reflection, menu):
    groups = {}
    for item in menu:
        group = groups.setdefault(item['name'], [])
        group.append({'binding': f'b{len(group)}', **{k: v for k, v in item.items() if k not in {'name', 'action_id'}}})
    for values in groups.values():
        if len(values) == 1:
            values[0].pop('binding')
    return {'task': task, 'plan': plan, 'last_reflection': reflection, 'available_bindings': groups}


def parse_native(message, menu):
    import jsonschema
    calls = message.get('tool_calls', [])
    if len(calls) != 1:
        raise ValueError('Select exactly one bound function call')
    function = calls[0]['function']
    definition = next(tool['function'] for tool in declarations(menu) if tool['function']['name'] == function['name'])
    parameters = json.loads(function['arguments'])
    jsonschema.validate(parameters, definition['parameters'])
    bound = [item for item in menu if item['name'] == function['name']]
    index = int(parameters['binding'][1:]) if len(bound) > 1 else 0
    return {'action_id': bound[index]['action_id']}, {'native_function': function['name'], 'native_binding': f'b{index}'}


def repair_message(message):
    return 'No tool has executed. The previous control response was invalid: '+json.dumps(message, ensure_ascii=False)+\
        '\nSelect exactly one available function. Use a listed local binding when the function has several bindings; otherwise use empty arguments.'


def select_native(client, task, plan, reflection, menu, budget, tag):
    import jsonschema
    tools = declarations(menu)
    messages = [{'role': 'system', 'content': NATIVE_PROMPT}, {'role': 'user', 'content': json.dumps(
        native_payload(task, plan, reflection, menu), ensure_ascii=False)}]
    for attempt in range(2):
        message = client.complete_tool(messages, tools, budget, tag=tag+('_repair' if attempt else ''))
        try:
            return parse_native(message, menu)
        except (ValueError, TypeError, KeyError, StopIteration, jsonschema.ValidationError) as error:
            if attempt:
                raise ValueError('Invalid native action selection after control repair') from error
            messages += [{'role': 'user', 'content': repair_message(message)}]
