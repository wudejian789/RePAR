"""Audit a completed frozen v4 run without inference or grading changes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import hashlib
import numpy as np

from repar.benchmarks.v3_suite import sha, unique, timestamp
from repar.benchmarks.math_exam import aggregate_grades, public_question, score_choice, solver_messages, task_text
from repar.evaluation import extract_answer, official_prompt, qa_text
from repar.io import digest, load_public_samples, public_item, read_json, read_jsonl, write_json
from repar.v4.knowledge import request_identity
from repar.v4.memory import source_boundaries, complete_word_end, merge_ranges, EvidenceMemory


def structured_request(run, calls, tag, check):
    """Recover the real task payload and validate any control repair separately."""
    attempts = [call for call in calls if call['tag'] in {tag, tag+'_repair'}]
    check('structured_attempt_order', [call['tag'] for call in attempts] in ([tag], [tag, tag+'_repair']))
    first = read_json(run/'cache/solver'/(attempts[0]['key']+'.json'))['request']
    accepted = attempts[-1]['response']['choices'][0]['message']['content']
    if len(attempts) == 2:
        repair = read_json(run/'cache/solver'/(attempts[1]['key']+'.json'))['request']
        previous = attempts[0]['response']['choices'][0]['message']['content']
        messages = first['messages'] + [{'role': 'assistant', 'content': previous},
            {'role': 'user', 'content': 'Return a complete concise object satisfying the schema.'}]
        check('repair_reuses_original_control_payload', repair == {**first, 'messages': messages})
    return first, json.loads(accepted)


def audit(run):
    run = Path(run)
    protocol = read_json(run/"protocol.json")
    config, solver = protocol["identity"]["config"], protocol["identity"]["solver"]
    checks = Counter()
    def check(label, value):
        if not value:
            raise AssertionError(label)
        checks[label] += 1
    for path, expected in protocol["identity"]["files"].items():
        check("frozen_source", sha(run/"source"/path) == expected)
    for category in ("data", "references"):
        for path, expected in protocol["identity"][category].items():
            check("unchanged_"+category, sha(path) == expected)
    batch_actions = solver['agent_version'] == 'react-v4-r27-hope2'
    alternative_actions = solver['agent_version'] == 'react-v4-r27-hope3'
    focused_action = batch_actions or alternative_actions
    action_query = solver['agent_version'] == 'react-v4-r27-hope' or focused_action
    seed_replica = config.get('historical_seed_reference')
    compare_r27 = action_query or bool(seed_replica)
    if compare_r27:
        baseline = Path('artifacts/repar_v4_r27_20261120')
        reference = read_json(baseline/'protocol.json')
        check('r27_data_matches', protocol['identity']['data'] == reference['identity']['data'])
        for path, checksum in reference['identity']['references'].items():
            if path.endswith('.pkl'):
                check('r27_index_matches', checksum == protocol['identity']['references'][path])
        if seed_replica:
            check('historical_replica_has_only_solver_seed_change', solver == {
                **reference['identity']['solver'], 'seed': solver['seed']} and solver['seed'] in (43, 44))
            check('historical_replica_reuses_frozen_r27_source',
                  all(protocol['identity']['files'].get(name) == checksum for name, checksum in reference['identity']['files'].items()))
            check('historical_replica_retains_evaluation_settings', all(
                config.get(key) == value for key, value in reference['identity']['config'].items()
                if key not in {'experiment_id', 'output', 'solver_config', 'report_filename'}))
        else:
            controller_changes = {'action_max_tokens'} if focused_action else set()
            for key, value in reference['identity']['solver'].items():
                if key not in {'agent_version', 'deployment_id', 'notes'} | controller_changes:
                    check('r27_solver_settings_match', solver[key] == value)
        check('r27_judge_settings_match', config['judge'] == reference['identity']['config']['judge'])
        for role in ('solver', 'judge'):
            expected = reference['deployments'][role]['props']
            actual = protocol['deployments'][role]['props']
            for key in ('model_alias', 'model_path', 'build_info', 'total_slots', 'chat_template',
                        'default_generation_settings'):
                check('r27_deployment_matches', actual[key] == expected[key])
    questions = {r["id"]: r for r in read_jsonl(config["math_questions"])}
    public_lb = {r["_id"]: public_item(r) for r in load_public_samples(config["longbench_data"])}
    labelled_lb = {r["_id"]: r for r in load_public_samples(config["longbench_data"])}
    context_by_hash = {}
    context_paths = {}
    for path in protocol["identity"]["references"]:
        if path.endswith('.pkl') and (not compare_r27 or path in reference['identity']['references']):
            stats = read_json(Path(path).with_suffix('.stats.json'))
            context_paths[stats['context_sha256']] = path
    diagnostics = {}
    for dataset, expected in (("math", questions), ("longbench", public_lb)):
        rows = unique(run/dataset/"predictions.jsonl", expected=expected)
        if compare_r27:
            original_rows = unique(baseline/dataset/'predictions.jsonl', expected=expected)
        check("complete_unique_coverage", len(rows) == (19 if dataset == "math" else 125))
        check("locked_predictions", sha(run/dataset/"predictions.jsonl") == read_json(run/dataset/"generation_manifest.json")["predictions_sha256"])
        diagnostics[dataset] = {"termination": dict(Counter(r.get("termination") for r in rows.values())),
            "no_evidence": sum(not r.get("evidence_ids") for r in rows.values()),
            "tools": dict(Counter(e["name"] for r in rows.values() for s in r["trace"] for e in s["executed"])),
            "control_errors": sum(r.get("termination") == "invalid_control" for r in rows.values()),
            "mean_evidence_sources": sum(len(r.get("evidence_ids", [])) for r in rows.values())/len(rows),
            "final_length": sum(c['response']['choices'][0]['finish_reason'] == 'length' for r in rows.values() for c in r['calls'] if c['tag'] in ('math_repar_answer', 'v2_repar_answer'))}
        if action_query:
            diagnostics[dataset]['blocked_actions'] = sum('blocked_before_execution' in step
                for row in rows.values() for step in row['trace'])
        if focused_action:
            diagnostics[dataset]['blocked_actions'] = sum(len(step.get('blocked_actions', []))
                for row in rows.values() for step in row['trace'])
        if alternative_actions:
            diagnostics[dataset]['fallback_calls'] = sum(bool(step['executed'] and step.get('blocked_actions'))
                for row in rows.values() for step in row['trace'])
            diagnostics[dataset]['unused_alternatives'] = sum(len(step.get('unused_alternatives', []))
                for row in rows.values() for step in row['trace'])
            diagnostics[dataset]['multi_action_rounds'] = sum(len(step['executed']) > 1
                for row in rows.values() for step in row['trace'])
        for qid, row in rows.items():
            check("correct_version_and_runtime_status", row["agent_version"] == solver["agent_version"] and row["status"] == "ok")
            if dataset == "math":
                check("public_task", row["task"] == task_text(questions[qid]) and row["question"] == public_question(questions[qid]))
            else:
                check("public_task", row["task"] == qa_text(public_lb[qid]))
                check("original_score", row["pred"] == extract_answer(row["response"]) and row["gold"] == labelled_lb[qid]["answer"] and row["correct"] == (row["pred"] == row["gold"]))
            ctx_hash = row["context_sha256"]
            if ctx_hash not in context_by_hash:
                with open(context_paths[ctx_hash], 'rb') as stream:
                    index = pickle.load(stream)
                context_by_hash = {ctx_hash: index.context}
            context = context_by_hash[ctx_hash]
            if compare_r27:
                reuse = row['index_reuse'] if dataset == 'longbench' else read_json(run/'math/index_reuse.json')
                check('r27_uses_historical_index', reuse['path'] == context_paths[ctx_hash] and
                      reuse['stats']['extractor']['model'] == 'en_core_web_sm')
            check("original_context", digest(context) == ctx_hash)
            if solver.get('source_similarity'):
                encoder_config = solver['source_similarity']
                manifest_path = Path(encoder_config['path'])/'download_manifest.json'
                check('auxiliary_encoder_manifest_identity', sha(manifest_path) == encoder_config['manifest_sha256'])
                manifest = read_json(manifest_path)
                stats = row['state']['source_similarity']
                check('recorded_auxiliary_encoder_identity', stats['model'] == manifest['model'] and stats['revision'] == manifest['revision'])
                check('encoder_does_not_generate_answers', stats['device'] == 'cpu' and stats['generative_model_calls'] == 0)
            if solver.get('source_reranker'):
                reranker = solver['source_reranker']
                manifest_path = Path(reranker['path'])/'download_manifest.json'
                check('reranker_manifest_matches_frozen_identity', sha(manifest_path) == reranker['manifest_sha256'])
                manifest = read_json(manifest_path)
                stats = row['state']['source_reranking']
                check('reranker_trace_identity', stats['model'] == manifest['model'] and stats['revision'] == manifest['revision'])
                check('reranker_resource_and_relevance_contract', stats['device'] == 'cpu' and stats['generative_model_calls'] == 0 and not stats['scores_are_entailment_probabilities'])
                if reranker.get('batch_by_length'):
                    check('reranker_declares_length_order', stats['batch_order'] == 'ascending_input_length')
                journal = stats.get('raw_pair_model_journal')
                check('reranker_journal_presence_matches_participation', bool(journal) == bool(stats['query_source_pairs']))
                if journal:
                    check('reranker_raw_journal_hash', sha(journal['path']) == journal['sha256'])
                    for batch in read_jsonl(journal['path']):
                        count = len(batch['query_source_pairs'])
                        groups, windows, logits = batch['query_window_feature_indices'], batch['window_inputs'], batch['window_logits']
                        check('reranker_raw_coverage', count == len(groups) == len(batch['relevance_scores']) and len(windows) == len(logits))
                        check('reranker_only_scored_original_source_text', all(text in context for _, text in batch['query_source_pairs']))
                        check('reranker_raw_window_capacity', all(0 < len(f['input_ids']) <= 512 for f in windows))
                        check('reranker_raw_logits_finite', all(np.isfinite(logits)))
                        reproduced = [sum(max(logits[i] for i in indices) for indices in group)/len(group) for group in groups]
                        check('reranker_pair_aggregation_reproducible', reproduced == batch['relevance_scores'])
                        if reranker.get('batch_by_length'):
                            order = [i for group in batch['batch_feature_indices'] for i in group]
                            check('length_batches_cover_all_recorded_windows', sorted(order) == list(range(len(windows))))
                            sizes = [len(windows[i]['input_ids']) for i in order]
                            check('length_batches_are_sorted', sizes == sorted(sizes))
            if solver.get('semantic_source_retrieval'):
                metadata = row['state']['semantic_source_index']
                check('semantic_source_cache_hash', sha(metadata['path']) == metadata['sha256'])
                identity = metadata['identity']
                check('semantic_cache_uses_original_context', identity['context_sha256'] == ctx_hash)
                spans = [list(span) for span in index.spans]
                check('semantic_cache_uses_original_partitions', identity['spans_sha256'] == hashlib.sha256(json.dumps(spans, separators=(',', ':')).encode()).hexdigest())
                with np.load(metadata['path'], allow_pickle=False) as stored:
                    check('semantic_cache_identity_matches_trace', json.loads(str(stored['identity'])) == identity)
                    check('semantic_cache_span_mapping', stored['spans'].tolist() == spans)
                    check('semantic_cache_vectors_finite', stored['vectors'].shape[0] == len(spans) and np.isfinite(stored['vectors']).all())
                check('semantic_cache_source_text_unchanged', not metadata['source_text_or_partitions_changed'])
            completed, observed, visible, base_queries = set(), {}, set(), set()
            known_entities = {}
            known_tree_previews = {}
            tracks_progress = solver['agent_version'] == 'react-v4-r27' or action_query
            tracks_source_references = tracks_progress or solver['agent_version'] in {'react-v4-r15', 'react-v4-r16', 'react-v4-r17', 'react-v4-r18', 'react-v4-r19', 'react-v4-r20', 'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'}
            tracks_view_queries = solver['agent_version'] in {'react-v4-r16', 'react-v4-r17', 'react-v4-r18', 'react-v4-r19', 'react-v4-r20', 'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'}
            for step in row["trace"]:
                if action_query and step.get('plan'):
                    from repar.v4.workflow import OBJECTIVE_ONLY_PLAN_SCHEMA
                    tag = f"v4_plan_{step['round']}"
                    request, plan = structured_request(run, row['calls'], tag, check)
                    check('plan_only_returns_objective', set(plan) == {'objective'} and plan == step['plan'])
                    check('plan_uses_objective_only_schema', request['response_format']['json_schema']['schema'] ==
                          OBJECTIVE_ONLY_PLAN_SCHEMA)
                    if proposal := step.get('action_proposal'):
                        tag = f"v4_action_{step['round']}"
                        request, action = structured_request(run, row['calls'], tag, check)
                        check('action_proposal_matches_response', action == proposal)
                        state = json.loads(request['messages'][-1]['content'])
                        check('action_sees_objective_and_bound_menu', state['plan'] == plan and
                              state['available_actions'] == step['available_actions'])
                        if focused_action:
                            from jsonschema import validate
                            from repar.v4.selection import FOCUSED_ACTION_ALTERNATIVES_PROMPT, FOCUSED_ACTION_BATCH_PROMPT
                            prompt = FOCUSED_ACTION_ALTERNATIVES_PROMPT if alternative_actions else FOCUSED_ACTION_BATCH_PROMPT
                            check('action_uses_focused_prompt', request['messages'][0]['content'] == prompt)
                            validate(action, request['response_format']['json_schema']['schema'])
                            remaining = solver['react_max_steps']-(step['round']-1 if alternative_actions else len(completed))
                            limit = solver['max_action_proposals'] if alternative_actions else min(remaining, solver['max_actions_per_round'])
                            check('action_uses_original_remaining_budget',
                                  1 <= len(action['actions']) <= limit
                                  and remaining == state['remaining_actions'])
                            if dataset == 'math':
                                task = json.loads(row['task'])
                                parts = [task['question_stem'], *(item['text'] for item in task['choices']),
                                         *(item['question'] for item in task['subquestions'])]
                            else:
                                item = public_lb[qid]
                                parts = [item['question'], *(item['choice_'+key] for key in 'ABCD')]
                            check('batch_action_sees_user_visible_task', state['task'] == '\n'.join(parts))
                            check('batch_menu_exposes_bindings_without_task_copy',
                                  all('anchor_query' not in a['parameters'] for a in state['available_actions']))
                            expected, blocked_calls, seen = [], [], set(completed)
                            for position, item in enumerate(action['actions']):
                                candidate = next(a for a in step['available_actions'] if a['action_id'] == item['action_id'])
                                params = dict(candidate['parameters'])
                                if 'query' in item:
                                    params['query'] = item['query'].strip()
                                    if (candidate['purpose'] == 'Combine independent focus and original-task search channels'
                                            and params['query'] != row['task']):
                                        params['anchor_query'] = row['task']
                                identity = request_identity(candidate['name'], params)
                                reason = None
                                if params.get('query') == '':
                                    reason = 'Search query must contain non-whitespace text'
                                elif identity in seen:
                                    reason = 'Duplicate completed action' if identity in completed else 'Duplicate action in batch'
                                if reason:
                                    blocked_calls.append({'proposal': item, 'reason': reason})
                                else:
                                    expected.append({'action_id': item['action_id'], 'tool': candidate['name'],
                                                     'parameters': params})
                                    seen.add(identity)
                                    if alternative_actions:
                                        check('alternatives_are_not_executed_without_feedback',
                                              step['unused_alternatives'] == action['actions'][position+1:])
                                        break
                            decisions = [step['decision']] if alternative_actions and step.get('decision') else step.get('decisions', [])
                            check('selection_preserves_model_order_and_bindings', decisions == expected)
                            check('batch_blocks_only_invalid_or_repeat_calls', step.get('blocked_actions', []) == blocked_calls)
                            if any(item['tool'] == 'answer' for item in expected):
                                check('answer_choice_is_exclusive_and_not_a_retrieval', len(expected) == 1 and not step['executed'])
                            else:
                                actual = [(e['name'], e['parameters']) for e in step['executed']]
                                check('batch_execution_matches_legal_calls', actual == [(d['tool'], d['parameters']) for d in expected])
                        elif blocked := step.get('blocked_before_execution'):
                            check('invalid_or_duplicate_action_was_not_executed', not step['executed'])
                            check('blocked_action_matches_proposal', blocked['proposal'] == action)
                if step.get('joint_control') and (step.get('decision') or step.get('blocked_before_execution')):
                    from repar.v4.planning import bind_plan, plan_request
                    plan_context = step['joint_control']
                    request, value = structured_request(run, row['calls'], f"v4_joint_plan_{step['round']}", check)
                    bound_scope = solver['agent_version'] in {'react-v4-r25', 'react-v4-r26'}
                    check('joint_plan_uses_frozen_search_scope_protocol',
                          plan_context.get('bind_search_scope', False) == bound_scope)
                    messages, schema = plan_request(plan_context['state'], plan_context['candidates'],
                                                    solver['retrieval_budget_options'], bind_search_scope=bound_scope)
                    check('joint_plan_uses_original_task_and_bound_menu', plan_context['state']['task'] == row['task'] and request['messages'] == messages)
                    check('joint_plan_uses_dynamic_schema', request['response_format']['json_schema']['schema'] == schema)
                    joint_plan, joint_call = bind_plan(value, plan_context['candidates'], row['task'])
                    check('joint_plan_matches_original_response', joint_plan == step['plan'])
                    if bound_scope:
                        for candidate in plan_context['candidates']:
                            if candidate['name'] == 'tree_search':
                                parameters = candidate['parameters']
                                scope = parameters['scope']
                                if 'scope_preview' in candidate:
                                    check('search_scope_preview_was_actually_observed',
                                          candidate['scope_preview'] == known_tree_previews.get(scope))
                        if joint_call['name'] == 'tree_search' and joint_call['parameters']['mode'] == 'tree':
                            check('tree_scope_preserves_its_original_query_contract',
                                  'anchor_query' not in joint_call['parameters'])
                    if step.get('duplicate_reselection'):
                        reselection = step['duplicate_reselection']
                        check('joint_duplicate_selection_matches_original_response', joint_call == reselection['rejected'])
                        check('joint_duplicate_selection_was_previously_executed',
                              request_identity(joint_call['name'], joint_call['parameters']) in completed)
                        check('reselection_menu_excludes_completed_actions', all(request_identity(a['name'], a['parameters']) not in completed
                              for a in reselection['available_actions']))
                        chosen = reselection['selected']
                        check('reselection_has_original_available_actions', reselection['available_actions'] == step['available_actions'])
                        calls = step['executed']
                        check('reselection_executes_alternative_once', not step.get('blocked_before_execution') and
                              (not calls if chosen['name'] == 'answer' else
                               len(calls) == 1 and calls[0]['name'] == chosen['name'] and calls[0]['parameters'] == chosen['parameters']))
                    elif step.get('blocked_before_execution'):
                        check('joint_duplicate_blocked_before_actual_execution', joint_call == step['blocked_before_execution'] and
                              request_identity(joint_call['name'], joint_call['parameters']) in completed and not step['executed'])
                    else:
                        check('joint_call_matches_actual_decision', step['available_actions'] == [joint_call])
                for candidate in step.get("available_actions", []):
                    check("completed_actions_not_selectable", request_identity(candidate["name"], candidate["parameters"]) not in completed)
                    if tracks_progress and 'scope_preview' in candidate:
                        scope = candidate['parameters']['scope']
                        check('search_scope_preview_was_actually_observed', candidate['scope_preview'] == known_tree_previews.get(scope))
                    if 'provenance_entity' in candidate:
                        entity = known_entities[candidate['provenance_entity']]
                        check('entity_origin_read_uses_actual_known_references', candidate['name'] == 'read' and candidate['parameters']['radius'] == 0 and set(candidate['parameters']['ids']) <= set(entity['passage_ids']))
                        check('entity_origin_read_capacity_bounded', len(candidate['parameters']['ids']) <= solver['retrieval_top_k'])
                    if 'cursor_query' in candidate:
                        params = candidate['parameters']
                        cid, current_round = params['id'], step['round']
                        available_stages = {f'controller_{current_round}'}
                        for earlier_round in range(1, current_round):
                            available_stages.update(f'{prefix}{earlier_round}{suffix}' for prefix, suffix in (
                                ('controller_', ''), ('observation_', ''),
                                ('observation_', '_retained'), ('observation_', '_propositions')))
                        ranges = [(span['start'], span['end']) for shown in row['state']['exposures']
                                  if shown['stage'] in available_stages for span in shown['spans'] if span['id'] == cid]
                        source = observed[cid]
                        cursor = params['offset']
                        end = cursor + params['length']
                        check('focused_cursor_stays_within_retrieved_source', source['start'] <= cursor < end <= source['end'])
                        check('focused_cursor_leaves_previous_views_intact',
                              all(end <= lo or hi <= cursor for lo, hi in merge_ranges(ranges)))
                        check('focused_cursor_starts_at_sentence_boundary',
                              cursor - source['start'] in source_boundaries(source['text']))
                if step.get("decision") and not alternative_actions and (not step.get('joint_control') or step.get('duplicate_reselection')):
                    choice = step["decision"]
                    selected = next(a for a in step["available_actions"] if a["action_id"] == choice["action_id"])
                    selection = step
                    if step.get('duplicate_reselection'):
                        reselection = step['duplicate_reselection']
                        selection = reselection['selection']
                        check('reselected_call_matches_actual_decision', selected == reselection['selected'])
                    parameters = dict(selected['parameters'])
                    if action_query and selected['name'] in {'tree_search', 'graph_entities', 'graph_neighbors'}:
                        parameters['query'] = step['action_proposal']['query'].strip()
                        if parameters.get('anchor_query') == parameters['query']:
                            parameters.pop('anchor_query')
                    check("choice_matches_bound_call", selected["name"] == choice["tool"] and parameters == choice["parameters"])
                    if solver.get('native_action_tools'):
                        from repar.v4.native import declarations, repair_message, parse_native, native_payload, NATIVE_PROMPT
                        from repar.v4.selection import family_menu, action_families
                        tag = f"v4_action_{step['round']}"
                        attempts = [call for call in row['calls'] if call['tag'] in {tag, tag+'_repair'}]
                        check('native_control_attempt_order', [call['tag'] for call in attempts] in ([tag], [tag, tag+'_repair']))
                        first = read_json(run/'cache/solver'/(attempts[0]['key']+'.json'))['request']
                        menu = family_menu(action_families(step['available_actions']))
                        check('native_declarations_match_actual_bound_menu', first['tools'] == declarations(menu))
                        previous = row['trace'][step['round']-2].get('reflection') if step['round'] > 1 else None
                        check('native_request_has_original_task_and_actual_bindings', first['messages'] == [
                            {'role': 'system', 'content': NATIVE_PROMPT}, {'role': 'user', 'content': json.dumps(
                                native_payload(row['task'], step['plan'], previous, menu), ensure_ascii=False)}])
                        if len(attempts) == 2:
                            repaired = read_json(run/'cache/solver'/(attempts[1]['key']+'.json'))['request']
                            old = attempts[0]['response']['choices'][0]['message']
                            check('native_repair_precedes_actual_execution', repaired == {**first, 'messages': first['messages']+[{'role': 'user', 'content': repair_message(old)}]})
                        message = attempts[-1]['response']['choices'][0]['message']
                        check('native_action_has_one_function', len(message['tool_calls']) == 1)
                        function = message['tool_calls'][0]['function']
                        action, binding = parse_native(message, menu)
                        check('native_name_matches_actual_operator', function['name'] == choice['tool'] == step['native_function'])
                        check('native_binding_has_only_operator_local_scope', binding['native_binding'] == step['native_binding'])
                    else:
                        _, action = structured_request(run, row['calls'], f"v4_action_{step['round']}", check)
                    model_choice = action['action_id']
                    if solver.get('factor_action_parameters'):
                        from repar.v4.selection import action_families, result_limit_payload, RESULT_LIMIT_PROMPT
                        families = action_families(step['available_actions'])
                        check('families_match_available_concrete_calls', families == selection['action_families'])
                        check('family_choice_matches_model', model_choice == selection['selected_family'])
                        family = next(f for f in families if f['action_id'] == model_choice)
                        variants = family['variants']
                        check('bound_variant_is_in_chosen_family', selected in variants)
                        check('resource_subdecision_only_when_needed', selection['resource_selection_model_call'] == (len(variants) > 1))
                        if len(variants) > 1:
                            request, binding = structured_request(run, row['calls'], f"v4_parameters_{step['round']}", check)
                            check('resource_choice_matches_model', binding['top_k'] == choice['parameters']['top_k'])
                            limits = [v['parameters']['top_k'] for v in variants]
                            per_query = variants[0].get('result_limit_scope') == 'each_query'
                            expected = [{'role': 'system', 'content': RESULT_LIMIT_PROMPT}, {'role': 'user',
                                'content': json.dumps(result_limit_payload(row['task'], step['plan']['objective'], limits, per_query=per_query), ensure_ascii=False)}]
                            check('resource_request_has_exact_legal_limits', request['messages'] == expected and request['response_format']['json_schema']['schema']['properties']['top_k']['enum'] == limits)
                        else:
                            check('singleton_variant_matches_exact_call', variants == [selected])
                    else:
                        check("choice_matches_model", model_choice == choice["action_id"])
                round_new, round_repeat = [], []
                if batch_actions:
                    round_references = {cid for entity in known_entities.values() for cid in entity.get('passage_ids', [])} - set(observed)
                for executed in step["executed"]:
                    identity = request_identity(executed["name"], executed["parameters"])
                    check("no_duplicate_execution", identity not in completed)
                    completed.add(identity)
                    observation = executed["observation"]
                    check("actual_tool_status", observation["status"] in ("ok", "no_match"))
                    if tracks_progress:
                        returned = [record['id'] for record in observation.get('results', [])]
                        new = [cid for cid in returned if cid not in observed]
                        repeated = [cid for cid in returned if cid in observed]
                        if batch_actions:
                            round_new.extend(new)
                            round_repeat.extend(repeated)
                        else:
                            check('progress_new_sources_match_tool_history', step['progress']['new_sources'] == new)
                            check('progress_repeat_sources_match_tool_history', step['progress']['previously_retrieved_sources'] == repeated)
                        if executed['name'] == 'graph_neighbors':
                            expected = index.execute('graph_neighbors', {**executed['parameters'], 'eligible_edges_first': True})
                            check('graph_path_replays_eligible_edges_first', expected['paths'] == observation['paths'])
                    if tracks_source_references and not batch_actions:
                        previous_references = {cid for entity in known_entities.values() for cid in entity.get('passage_ids', [])} - set(observed)
                    for record in observation.get("results", []):
                        check("observed_exact_original_span", context[record["start"]:record["end"]] == record["text"])
                        observed[record["id"]] = record
                        known_entities.update({entity['id']: entity for entity in record.get('entities', [])})
                        if solver['agent_version'] in {'react-v4-r6', 'react-v4-r7', 'react-v4-r8', 'react-v4-r9', 'react-v4-r10', 'react-v4-r11', 'react-v4-r12', 'react-v4-r13', 'react-v4-r14', 'react-v4-r15', 'react-v4-r16', 'react-v4-r17', 'react-v4-r18', 'react-v4-r19', 'react-v4-r20', 'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'} and 'entity_descriptor_selection' in record:
                            selected = record['entity_descriptor_selection']
                            native_ids = index.entity_graph.passage_entities[int(record['id'][1:])]
                            check('source_entity_candidate_ids_are_original', selected['candidate_ids'] == native_ids == record['entity_ids'])
                            check('source_entity_descriptors_are_bounded', len(record['entities']) == min(24, len(native_ids)))
                            check('source_entity_selection_is_returned', selected['selected_ids'] == [e['id'] for e in record['entities']])
                            for entity in record['entities']:
                                check('source_entity_description_is_original', entity == index.entity_graph.describe(entity['id']))
                                check('source_entity_has_actual_source_link', record['id'] in index.entity_graph.entity(entity['id'])['passage_ids'])
                    known_entities.update({entity['id']: entity for entity in observation.get('entities', [])})
                    if tracks_source_references and not batch_actions:
                        references = {cid for entity in known_entities.values() for cid in entity.get('passage_ids', [])} - set(observed)
                        check('only_new_unread_entity_references_count_as_progress',
                              step['progress']['new_source_references'] == sorted(references-previous_references))
                    if solver.get('read_entity_sources'):
                        for entity in executed['visible_observation']['entities']:
                            check('entity_source_metadata_is_original', entity.get('passage_ids') == known_entities[entity['id']]['passage_ids'])
                    if executed["name"] == "tree_search":
                        if solver.get('retrieval_budget_options'):
                            check('model_selected_a_configured_retrieval_budget', executed['parameters']['top_k'] in solver['retrieval_budget_options'])
                        for channel in observation["query_channels"]:
                            expected_query = executed["parameters"]["query"] if channel["channel"] == "focus" else executed["parameters"]["anchor_query"]
                            check("independent_query_channels", channel["parameters"]["query"] == expected_query)
                        check("no_zero_match_padding", all(r["score"] > 0 for r in observation["results"]))
                        channels = observation["query_channels"]
                        if (tracks_progress or solver["agent_version"] in {"react-v4-r2", "react-v4-r3", "react-v4-r4", "react-v4-r5", "react-v4-r6", "react-v4-r7", "react-v4-r8", "react-v4-r9", "react-v4-r10", "react-v4-r11", "react-v4-r12", "react-v4-r13", "react-v4-r14", "react-v4-r15", "react-v4-r16", "react-v4-r17", "react-v4-r18", "react-v4-r19", "react-v4-r20", "react-v4-r22", "react-v4-r23", "react-v4-r24", "react-v4-r25", "react-v4-r26"}) and len(channels) == 2:
                            scores = {}
                            for channel in channels:
                                for rank, cid in enumerate(channel["ids"]):
                                    scores[cid] = scores.get(cid, 0)+1/(21+rank)
                            limit = executed['parameters']['top_k']
                            per_query = solver['agent_version'] in {'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'}
                            ranked = sorted(scores, key=lambda cid: (-scores[cid], index.spans[int(cid[1:])][0]))
                            if per_query:
                                check('cap_applies_to_each_actual_query', observation['fusion']['result_limit_scope'] == 'each_query' and
                                      all(c['parameters']['top_k'] == limit and len(c['ids']) <= limit for c in channels))
                            else:
                                ranked = ranked[:limit]
                            check("actual_equal_rank_fusion", [r["id"] for r in observation["results"]] == ranked)
                            check("query_channel_weights", observation["fusion"]["channel_weights"] == {"focus": 1, "task": 1})
                            if tracks_source_references and not tracks_progress and not per_query:
                                check('merge_channels_before_result_limit', observation['fusion']['requested_result_limit'] == limit and
                                      all(c['parameters']['top_k'] == max(limit, solver.get('retrieval_candidate_top_k', 8)) for c in channels))
                                check('reflection_only_receives_returned_source_ids', all(set(c['ids']) <= set(ranked)
                                      for c in executed['visible_observation']['query_channels']))
                            if tracks_view_queries:
                                bindings = observation['source_view_query_bindings']
                                check('view_queries_are_bound_only_to_returned_sources', set(bindings) == set(ranked))
                                for cid in ranked:
                                    position, channel = min(((c['ids'].index(cid), c) for c in channels if cid in c['ids']),
                                                            key=lambda pair: pair[0])
                                    check('view_query_is_bound_to_actual_best_ranked_channel', bindings[cid] == {
                                        'rank': position, 'channel': channel['channel'], 'query': channel['parameters']['query']})
                        if tracks_progress or solver["agent_version"] in {"react-v4-r3", "react-v4-r4", "react-v4-r5", "react-v4-r6", "react-v4-r7", "react-v4-r8", "react-v4-r9", "react-v4-r10", "react-v4-r11", "react-v4-r12", "react-v4-r13", "react-v4-r14", "react-v4-r15", "react-v4-r16", "react-v4-r17", "react-v4-r18", "react-v4-r19", "react-v4-r20", "react-v4-r22", "react-v4-r23", "react-v4-r24", "react-v4-r25", "react-v4-r26"}:
                            events = observation['query_execution']
                            check('base_query_events_match_channels', len(events) == len(channels))
                            for event, channel in zip(events, channels):
                                identity = request_identity('tree_search', channel['parameters'])
                                check('base_query_identity', identity == event['identity'])
                                check('base_query_reuse_matches_history', event['cached'] == (identity in base_queries))
                                base_queries.add(identity)
                                if tracks_progress:
                                    original = index.execute('tree_search', channel['parameters'])
                                    check('base_query_keeps_r3_tree_results',
                                          event['source_ids'] == [record['id'] for record in original['results']])
                                if solver['agent_version'] in {'react-v4-r5', 'react-v4-r6', 'react-v4-r7', 'react-v4-r8', 'react-v4-r9', 'react-v4-r10', 'react-v4-r11', 'react-v4-r12', 'react-v4-r13', 'react-v4-r14', 'react-v4-r15', 'react-v4-r16', 'react-v4-r17', 'react-v4-r18', 'react-v4-r19', 'react-v4-r20', 'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'} and channel['parameters'].get('mode', 'hybrid') == 'hybrid':
                                    generation = event['candidate_generation']
                                    p = channel['parameters']
                                    reranking = generation.get('pair_reranking')
                                    pool_limit = max(p['top_k'], solver['retrieval_candidate_top_k']) if reranking else p['top_k']
                                    lexical = [f'C{i}' for i, score in index.bm25(p['query'], pool_limit, p.get('scope')) if score > 0]
                                    routed = [r['id'] for r in index.execute('tree_search', {**p, 'top_k': pool_limit})['results'] if r['score'] > 0]
                                    check('full_scoped_lexical_candidates', generation['lexical_ids'] == lexical)
                                    check('original_routed_candidates', generation['routed_ids'] == routed)
                                    fused = {}
                                    for ids in (routed, lexical):
                                        for rank, cid in enumerate(ids):
                                            fused[cid] = fused.get(cid, 0)+1/(21+rank)
                                    selected = sorted(fused, key=lambda cid: (-fused[cid], index.spans[int(cid[1:])][0]))[:pool_limit]
                                    check('actual_candidate_fusion', selected == generation['selected_ids'])
                                    semantic = generation.get('semantic_channel')
                                    if semantic:
                                        check('semantic_channel_starts_from_lexical_tree_pool', selected == semantic['pre_semantic_ids'])
                                        node = index.nodes[index._scope(p.get('scope'))]
                                        check('semantic_candidates_within_scope', all(node['lo'] <= int(cid[1:]) < node['hi'] for cid in semantic['semantic_ids']))
                                        check('semantic_scores_reported', len(semantic['semantic_ids']) == len(semantic['cosine_scores']) and all(np.isfinite(v) for v in semantic['cosine_scores']))
                                        scores = {}
                                        for ids in (selected, semantic['semantic_ids']):
                                            for rank, cid in enumerate(ids):
                                                scores[cid] = scores.get(cid, 0)+1/(21+rank)
                                        merged = sorted(scores, key=lambda cid: (-scores[cid], index.spans[int(cid[1:])][0]))[:pool_limit]
                                        check('actual_semantic_candidate_fusion', merged == semantic['selected_ids'])
                                        if not reranking:
                                            check('semantic_event_matches_candidates', merged == event['source_ids'])
                                    elif not reranking:
                                        check('lexical_tree_event_matches_candidates', selected == event['source_ids'])
                                    if reranking:
                                        for rank, cid in enumerate(semantic['semantic_ids'] if semantic else []):
                                            fused[cid] = fused.get(cid, 0)+1/(21+rank)
                                        candidates = sorted(fused, key=lambda cid: index.spans[int(cid[1:])][0])
                                        values = reranking['relevance_logits']
                                        check('pair_reranking_preserves_scoped_channel_union', reranking['candidate_ids'] == candidates)
                                        check('pair_reranking_has_finite_logits_for_each_candidate', len(values) == len(candidates) and all(np.isfinite(values)))
                                        check('pair_reranking_distinguishes_pool_and_return_limit', reranking['candidate_channel_limit'] == pool_limit and reranking['requested_result_limit'] == p['top_k'])
                                        check('raw_relevance_not_entailment', reranking['scores_are_entailment_probabilities'] is False)
                                        ordered = sorted(range(len(candidates)), key=lambda i: (-values[i], -fused[candidates[i]], index.spans[int(candidates[i][1:])][0]))[:p['top_k']]
                                        check('pair_reranking_output_matches_logged_logits', [candidates[i] for i in ordered] == reranking['selected_ids'] == event['source_ids'])
                    for view in executed["visible_observation"]["source_views"]:
                        check("visible_exact_original_span", context[view["start"]:view["end"]] == view["text"])
                        visible.add(view["id"])
                    if observation['operator'] == 'tree_browse':
                        known_tree_previews.update({node['id']: node['preview']
                            for node in executed['visible_observation']['children'] if 'preview' in node})
                if batch_actions and step['executed']:
                    check('batch_progress_matches_tool_history', step['progress']['new_sources'] == round_new and
                          step['progress']['previously_retrieved_sources'] == list(dict.fromkeys(round_repeat)))
                    references = {cid for entity in known_entities.values() for cid in entity.get('passage_ids', [])} - set(observed)
                    check('batch_counts_only_new_unread_entity_references',
                          step['progress']['new_source_references'] == sorted(references-round_references))
                if "reflection" in step:
                    check("reflection_has_actual_observation", bool(step["executed"]))
                    check("reflection_cites_visible_sources", set(step["reflection"]["supported_sources"]) <= visible | {""})
                    request, reflection = structured_request(run, row['calls'], f"v4_reflection_{step['round']}", check)
                    check('reflection_matches_original_model_output', step['reflection'] == reflection)
                    payload = json.loads(request['messages'][-1]['content'])
                    if tracks_progress:
                        expected = {k: v for k, v in step['progress'].items() if k != 'new_source_references'}
                        expected['new_source_reference_count'] = len(step['progress']['new_source_references'])
                        check('reflection_gets_actual_navigation_and_evidence_progress', payload['observation_progress'] == expected)
                    if solver.get('reflection_protocol_examples'):
                        from repar.v4.protocol import reflection_messages
                        check('abstract_demonstrations_marked_and_separate', request['messages'] == reflection_messages(payload, True))
                        check('demonstrations_not_task_evidence', not row['state']['reflection_protocol_examples']['demonstrations_are_task_evidence'])
                    if batch_actions:
                        check('batch_reflection_gets_actual_actions', [
                            {'action_id': item['action_id'], 'tool': item['name'], 'parameters': item['parameters']}
                            for item in payload['actions']] == step['decisions'])
                        check('batch_reflection_gets_actual_observations', [
                            {k: v for k, v in item.items() if k != 'existing_sources'}
                            for item in payload['observations']] == [e['visible_observation'] for e in step['executed']])
                        check('batch_reflection_receives_blocked_calls', payload.get('blocked_actions', []) == step.get('blocked_actions', []))
                    else:
                        check("actual_visible_observation_sent", payload["observation"] == step["executed"][0]["visible_observation"])
                    if tracks_progress or solver['agent_version'] in {'react-v4-r3', 'react-v4-r4', 'react-v4-r5', 'react-v4-r6', 'react-v4-r7', 'react-v4-r8', 'react-v4-r9', 'react-v4-r10', 'react-v4-r11', 'react-v4-r12', 'react-v4-r13', 'react-v4-r14', 'react-v4-r15', 'react-v4-r16', 'react-v4-r17', 'react-v4-r18', 'react-v4-r19', 'react-v4-r20', 'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'}:
                        feedback = step['reflection_context']
                        check('actual_retained_source_context_sent', all(payload[k] == v for k, v in feedback.items()))
                        observations = feedback['observations'] if batch_actions else [
                            {**feedback['observation'], 'existing_sources': feedback['existing_sources']}]
                        current_views = [r for item in observations for r in item['source_views']]
                        retained_views = [r for item in observations for r in item['existing_sources']]
                        citable = {r['id'] for r in current_views+retained_views}
                        for record in retained_views:
                            check('retained_context_exact_original_span', context[record['start']:record['end']] == record['text'])
                        for p in (p for item in observations for p in item.get('propositions', [])):
                            parent = observed.get(p.get('passage_id'))
                            if parent and p.get('text') and p['text'] in parent['text']:
                                citable.add(parent['id'])
                        check('citation_ids_have_text_in_actual_request', citable == set(payload['available_sources']))
                        check('reflection_only_cites_current_payload', set(step['reflection']['supported_sources']) <= citable | {''})
                        check('one_budget_for_current_and_retained_views', feedback['source_view_tokens'] <= feedback['source_view_budget'] == solver['observation_evidence_tokens'])
                        if solver['agent_version'] in {'react-v4-r17', 'react-v4-r18', 'react-v4-r19', 'react-v4-r20', 'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'}:
                            check('prior_ranges_omit_currently_covered_spans', not any(
                                now['id'] == past['id'] and now['start'] <= past['start'] and past['end'] <= now['end']
                                for past in retained_views for now in current_views))
            if focused_action:
                check('batch_respects_task_tool_budget', len(completed) <= solver['react_max_steps'])
                check('batch_completed_calls_match_execution', sorted(completed) == row['state']['completed_actions'])
            for record in row["evidence_records"]:
                check("final_original_span", context[record["start"]:record["end"]] == record["text"])
                parent = observed[record["id"]]
                check("final_evidence_was_retrieved", parent["start"] <= record["start"] < record["end"] <= parent["end"])
                if tracks_progress or solver["agent_version"] in {"react-v4-r2", "react-v4-r3", "react-v4-r4", "react-v4-r5", "react-v4-r6", "react-v4-r7", "react-v4-r8", "react-v4-r9", "react-v4-r10", "react-v4-r11", "react-v4-r12", "react-v4-r13", "react-v4-r14", "react-v4-r15", "react-v4-r16", "react-v4-r17", "react-v4-r18", "react-v4-r19", "react-v4-r20", "react-v4-r22", "react-v4-r23", "react-v4-r24", "react-v4-r25", "react-v4-r26"}:
                    boundaries = {parent["start"]+b for b in source_boundaries(parent["text"])}
                    if solver['agent_version'] not in {'react-v4-r4', 'react-v4-r5', 'react-v4-r6', 'react-v4-r7', 'react-v4-r8', 'react-v4-r9', 'react-v4-r10', 'react-v4-r11', 'react-v4-r12', 'react-v4-r13', 'react-v4-r14', 'react-v4-r15', 'react-v4-r16', 'react-v4-r17', 'react-v4-r18', 'react-v4-r19', 'react-v4-r20', 'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'}:
                        check("automatic_window_starts_at_source_boundary", record["start"] in boundaries)
                    check("partial_sentence_marked", record["partial_sentence"] == (record["start"] not in boundaries or record["end"] not in boundaries))
                    if solver['agent_version'] not in {'react-v4-r4', 'react-v4-r5', 'react-v4-r6', 'react-v4-r7', 'react-v4-r8', 'react-v4-r9', 'react-v4-r10', 'react-v4-r11', 'react-v4-r12', 'react-v4-r13', 'react-v4-r14', 'react-v4-r15', 'react-v4-r16', 'react-v4-r17', 'react-v4-r18', 'react-v4-r19', 'react-v4-r20', 'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'}:
                        check("automatic_window_preserves_whole_word_end", complete_word_end(parent["text"], record["end"]-parent["start"]) == record["end"]-parent["start"])
            if solver['agent_version'] in {'react-v4-r4', 'react-v4-r5', 'react-v4-r6', 'react-v4-r7', 'react-v4-r8', 'react-v4-r9', 'react-v4-r10', 'react-v4-r11', 'react-v4-r12', 'react-v4-r13', 'react-v4-r14', 'react-v4-r15', 'react-v4-r16', 'react-v4-r17', 'react-v4-r18', 'react-v4-r19', 'react-v4-r20', 'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'}:
                for selection in row['state']['source_view_selections']:
                    check('stable_source_view_budget', selection['tokens'] <= selection['budget'])
                    if solver['agent_version'] in {'react-v4-r6', 'react-v4-r7', 'react-v4-r8', 'react-v4-r9', 'react-v4-r10', 'react-v4-r11', 'react-v4-r12', 'react-v4-r13', 'react-v4-r14', 'react-v4-r15', 'react-v4-r16', 'react-v4-r17', 'react-v4-r18', 'react-v4-r19', 'react-v4-r20', 'react-v4-r22', 'react-v4-r23', 'react-v4-r24', 'react-v4-r25', 'react-v4-r26'}:
                        for anchor in selection['anchor_ranges']:
                            check('context_restoration_preserves_selected_anchor', any(v['id'] == anchor['id'] and v['start'] <= anchor['start'] and anchor['end'] <= v['end'] for v in selection['selected_ranges']))
                    for kind in ('selected_ranges', 'retained', 'dropped_for_budget'):
                        for span in selection[kind]:
                            parent = observed[span['id']]
                            check('stable_source_view_original_bounds', parent['start'] <= span['start'] < span['end'] <= parent['end'])
            for exposure in row["state"]["exposures"]:
                for span in exposure["spans"]:
                    parent = observed[span["id"]]
                    check("exposure_ledger_original_bounds", parent["start"] <= span["start"] < span["end"] <= parent["end"])
            final = [c for c in row["calls"] if c["tag"] in ("math_repar_answer", "v2_repar_answer")]
            check("one_raw_final_generation", len(final) == 1 and final[0]["response"]["choices"][0]["message"]["content"] == row["response"])
            cached = read_json(run/"cache/solver"/(final[0]["key"]+".json"))
            if compare_r27:
                prior_key = original_rows[qid]['calls'][-1]['key']
                prior_body = read_json(baseline/'cache/solver'/f'{prior_key}.json')['request']
                check('r27_final_request_settings_match', cached['request'] == {
                    **prior_body, 'messages': cached['request']['messages'], 'seed': solver['seed'],
                })
            evidence = "\n\n".join(EvidenceMemory.block(r) for r in row["evidence_records"])
            expected_messages = solver_messages(row["task"], evidence or "未取得相关教材内容。") if dataset == "math" else [
                {"role": "user", "content": official_prompt(public_lb[qid], evidence or "No source evidence was obtained.", rag=True)}]
            check("unchanged_final_prompt", cached["request"]["messages"] == expected_messages)
            check("unchanged_final_budget", cached["request"]["max_tokens"] == (4096 if dataset == "math" else 128))
            for call in row["calls"]:
                record = read_json(run/"cache/solver"/(call["key"]+".json"))
                check("raw_response_matches_cache", record["response"] == call["response"])
                key = digest(json.dumps({"request": record["request"], "deployment_id": solver["deployment_id"]}, sort_keys=True, ensure_ascii=False))
                check("deployment_and_payload_cache_identity", key == call["key"])
                p = record["request"]
                check("original_model_generation", p["model"] == solver["model"] and p["seed"] == solver["seed"] and p["temperature"] == .1 and not p["chat_template_kwargs"]["enable_thinking"])
        if dataset == "math":
            for qid, grade in unique(run/"math/grades.jsonl", expected=questions).items():
                check("grade_response_identity", grade["response_sha256"] == digest(rows[qid]["response"]))
                recomputed = score_choice(questions[qid], rows[qid]["response"]) if questions[qid]["question_type"] in ("single_choice", "multiple_choice") else aggregate_grades(questions[qid], grade["grades"])
                check("unchanged_grade_aggregation", all(recomputed[k] == grade[k] for k in ("score", "fully_correct", "strict_score", "max_score")))
                for call in grade["judge_calls"]:
                    p = read_json(run/"cache/judge"/(call["key"]+".json"))["request"]
                    check("anonymous_judge_role", p["model"] == config["judge"]["model"] and p["temperature"] == 0 and p["seed"] in config["judge"]["replicate_seeds"])
    result = {"passed": True, "completed_at": timestamp(), "checks": dict(checks), "check_count": sum(checks.values()),
              "diagnostics": diagnostics, "no_inference_or_score_changes": True}
    write_json(run/"audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="artifacts/repar_v4_20261119")
    audit(parser.parse_args().run)
