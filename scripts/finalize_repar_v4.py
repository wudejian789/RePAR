"""Summarize and audit the full v4 condition; never modify predictions or grades."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import hashlib
import math
from pathlib import Path
import shutil
from statistics import mean
from zoneinfo import ZoneInfo

from audit_repar_v4 import audit
from repar.benchmarks.v3_suite import sha, unique
from repar.io import read_json, read_jsonl, write_json


def mapped(path):
    values = read_jsonl(path)
    result = {r.get('id', r.get('_id')): r for r in values}
    if len(result) != len(values):
        raise ValueError(f"Duplicate IDs: {path}")
    return result


def paired(a, b):
    good = lambda r: bool(r.get('correct', r.get('judge')))
    assert set(a) == set(b)
    wins = [qid for qid in a if good(a[qid]) and not good(b[qid])]
    losses = [qid for qid in a if not good(a[qid]) and good(b[qid])]
    n = len(wins)+len(losses)
    p = min(1, 2*sum(math.comb(n, i) for i in range(min(len(wins), len(losses))+1))/2**n) if n else 1
    return {'new_only_correct': wins, 'old_only_correct': losses, 'exact_mcnemar_p_unadjusted': p}


def finalize(run):
    run = Path(run)
    if (run/'completion.json').exists():
        completed = read_json(run/'completion.json')
        if not completed.get('complete'):
            raise ValueError('An existing non-complete manifest must not be overwritten')
        for name, expected in completed['files_sha256'].items():
            if sha(run/name) != expected:
                raise ValueError(f'Completed output changed: {name}')
        if sha(run/'protocol.json') != completed['frozen_generation']['protocol_sha256']:
            raise ValueError('Completed generation protocol changed')
        print(json.dumps({'already_complete': True, 'no_files_rewritten': True, 'run': str(run)}))
        return read_json(run/'analysis.json')
    summary = read_json(run/'summary.json')
    audited = audit(run)
    protocol = read_json(run/'protocol.json')
    identity = protocol['identity']
    config = identity['config']
    solver = identity['solver']
    new_name = config.get('comparison_label', 'v4')
    frozen_summary = {'sha256': sha(run/'summary.json'), 'protocol_sha256': sha(run/'protocol.json')}
    paths = {
        'direct': ('artifacts/math_2025_national_i_v5/direct_grades.jsonl', 'artifacts/experiment/direct.jsonl'),
        'bm25': ('artifacts/math_2025_national_i_v5/bm25_grades.jsonl', 'artifacts/experiment/bm25.jsonl'),
        'v2': ('artifacts/math_2025_national_i_v5/repar_grades.jsonl', 'artifacts/experiment_v2/repar.jsonl'),
        'v3': ('artifacts/repar_v3_core_r1_20261119/math/grades.jsonl', 'artifacts/repar_v3_core_r1_20261119/longbench/predictions.jsonl'),
        new_name: (run/'math/grades.jsonl', run/'longbench/predictions.jsonl'),
    }
    if new_name != 'v4':
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4原版': ('artifacts/repar_v4_20261119/math/grades.jsonl', 'artifacts/repar_v4_20261119/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r3', 'v4-r4', 'v4-r5', 'v4-r6', 'v4-r7+E5', 'v4-r8+E5', 'v4-r9+E5', 'v4-r10+E5', 'v4-r11+E5+pairrank', 'v4-r12+E5+pairrank', 'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r2': ('artifacts/repar_v4_r2_20261119/math/grades.jsonl', 'artifacts/repar_v4_r2_20261119/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r4', 'v4-r5', 'v4-r6', 'v4-r7+E5', 'v4-r8+E5', 'v4-r9+E5', 'v4-r10+E5', 'v4-r11+E5+pairrank', 'v4-r12+E5+pairrank', 'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r3': ('artifacts/repar_v4_r3_20261120/math/grades.jsonl', 'artifacts/repar_v4_r3_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r5', 'v4-r6', 'v4-r7+E5', 'v4-r8+E5', 'v4-r9+E5', 'v4-r10+E5', 'v4-r11+E5+pairrank', 'v4-r12+E5+pairrank', 'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r4': ('artifacts/repar_v4_r4_20261120/math/grades.jsonl', 'artifacts/repar_v4_r4_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r6', 'v4-r7+E5', 'v4-r8+E5', 'v4-r9+E5', 'v4-r10+E5', 'v4-r11+E5+pairrank', 'v4-r12+E5+pairrank', 'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r5': ('artifacts/repar_v4_r5_20261120/math/grades.jsonl', 'artifacts/repar_v4_r5_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r7+E5', 'v4-r8+E5', 'v4-r9+E5', 'v4-r10+E5', 'v4-r11+E5+pairrank', 'v4-r12+E5+pairrank', 'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r6': ('artifacts/repar_v4_r6_20261120/math/grades.jsonl', 'artifacts/repar_v4_r6_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r8+E5', 'v4-r9+E5', 'v4-r10+E5', 'v4-r11+E5+pairrank', 'v4-r12+E5+pairrank', 'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r7+E5': ('artifacts/repar_v4_r7_20261120/math/grades.jsonl', 'artifacts/repar_v4_r7_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r9+E5', 'v4-r10+E5', 'v4-r11+E5+pairrank', 'v4-r12+E5+pairrank', 'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r8+E5': ('artifacts/repar_v4_r8_20261120/math/grades.jsonl', 'artifacts/repar_v4_r8_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r10+E5', 'v4-r11+E5+pairrank', 'v4-r12+E5+pairrank', 'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r9+E5': ('artifacts/repar_v4_r9_20261120/math/grades.jsonl', 'artifacts/repar_v4_r9_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r11+E5+pairrank', 'v4-r12+E5+pairrank', 'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r10+E5': ('artifacts/repar_v4_r10_20261120/math/grades.jsonl', 'artifacts/repar_v4_r10_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r12+E5+pairrank', 'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r11+E5+pairrank': ('artifacts/repar_v4_r11_20261120/math/grades.jsonl', 'artifacts/repar_v4_r11_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r13+E5+pairrank', 'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r12+E5+pairrank': ('artifacts/repar_v4_r12_20261120/math/grades.jsonl', 'artifacts/repar_v4_r12_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name in {'v4-r14+E5+pairrank', 'v4-r15+E5+pairrank', 'v4-r16+E5+pairrank', 'v4-r17+E5+pairrank', 'v4-r18+E5+pairrank', 'v4-r19+E5', 'v4-r20+E5', 'v4-r21+E5', 'v4-r22+E5', 'v4-r23+E5', 'v4-r24+E5', 'v4-r25+E5', 'v4-r26+E5'}:
        paths = {**{k: v for k, v in paths.items() if k != new_name},
            'v4-r13+E5+pairrank': ('artifacts/repar_v4_r13_20261120/math/grades.jsonl', 'artifacts/repar_v4_r13_20261120/longbench/predictions.jsonl'),
            new_name: paths[new_name]}
    if new_name == 'v4-r27':
        recent = {
            name: (Path(f'artifacts/repar_v4_{name.removeprefix("v4-")}_20261120/math/grades.jsonl'),
                   Path(f'artifacts/repar_v4_{name.removeprefix("v4-")}_20261120/longbench/predictions.jsonl'))
            for name in ('v4-r3', 'v4-r25', 'v4-r26')
        }
        paths = {**{k: v for k, v in paths.items() if k != new_name}, **recent, new_name: paths[new_name]}
    target_methods = ('v2', 'v3')
    if config.get('comparison_runs'):
        paths = {}
        same_model_methods = {}
        for name, folder in config['comparison_runs'].items():
            reference = Path(folder)
            reference_files = ('protocol.json', 'audit.json', 'math/grades.jsonl', 'longbench/predictions.jsonl')
            for file in reference_files:
                source = reference / file
                if sha(source) != identity['references'].get(source.as_posix()):
                    raise ValueError(f'Comparison source differs from the frozen reference: {source}')
            old_protocol = read_json(reference/'protocol.json')
            if old_protocol['identity']['data'] != identity['data'] or not read_json(reference/'audit.json')['passed']:
                raise ValueError(f'Incompatible comparison condition: {reference}')
            old_solver = old_protocol['identity']['solver']
            if old_solver['model'] == solver['model']:
                same_model_methods[old_solver['agent_version']] = name
            paths[name] = (reference/'math/grades.jsonl', reference/'longbench/predictions.jsonl')
        paths[new_name] = (run/'math/grades.jsonl', run/'longbench/predictions.jsonl')
        target_methods = tuple(same_model_methods[version] for version in ('react-v2', 'react-v3-core-r1'))
    comparisons, all_math, all_lb, source_hashes = {}, {}, {}, {}
    for name, (math_path, lb_path) in paths.items():
        grades, lb = mapped(math_path), mapped(lb_path)
        assert len(grades) == 19 and len(lb) <= 125
        full_lb = len(lb) == 125
        if not full_lb:
            if name == new_name:
                raise ValueError('The evaluated condition must have all125 predictions')
            state_path = Path(lb_path).parent.parent/'execution_state.json'
            state = read_json(state_path)
            status = state.get('status')
            if status not in {'development_run_interrupted_resumable', 'math_complete_LB_not_started'}:
                raise ValueError(f'Incomplete reference lacks explicit development status: {lb_path}')
            if status == 'math_complete_LB_not_started' and lb:
                raise ValueError(f'Not-started reference has a prediction prefix: {lb_path}')
            # Archive the exact prefix so later resumption cannot change this
            # report's historical evidence. It is never scored as a full run.
            archive = run/'development_prefixes'/name/'predictions.jsonl'
            archive.parent.mkdir(parents=True, exist_ok=True)
            expected_prefix = sha(lb_path) if Path(lb_path).exists() else hashlib.sha256(b'').hexdigest()
            if not archive.exists():
                if Path(lb_path).exists():
                    shutil.copyfile(lb_path, archive)
                else:
                    archive.write_bytes(b'')
            elif sha(archive) != expected_prefix:
                raise ValueError('A previously archived development prefix changed')
            source_hashes[str(archive)] = sha(archive)
        all_math[name], all_lb[name] = grades, lb
        source_hashes[str(math_path)] = sha(math_path)
        source_hashes[str(lb_path)] = sha(lb_path) if Path(lb_path).exists() else None
        comparisons[name] = {'math_score': sum(r['score'] for r in grades.values()),
            'math_fully_correct': sum(r['fully_correct'] for r in grades.values()),
            'longbench_correct': sum(bool(r.get('correct', r.get('judge'))) for r in lb.values()) if full_lb else None,
            'longbench_completed': len(lb), 'longbench_complete': full_lb,
            'measurement': 'new frozen full run' if name == new_name else ('historical results reused, not rerun' if full_lb else 'full MATH; interrupted LongBench development prefix, no full accuracy')}
    pairs = {name: paired(all_lb[new_name], lb) for name, lb in all_lb.items() if name != new_name and len(lb) == 125}
    assert all(set(data) == set(all_math[new_name]) for data in all_math.values())
    assert all(set(data) <= set(all_lb[new_name]) for data in all_lb.values())
    details = []
    for qid in all_math[new_name]:
        details.append({'dataset': 'math', 'id': qid, **{name: data[qid]['score'] for name, data in all_math.items()}})
    for qid in all_lb[new_name]:
        details.append({'dataset': 'longbench', 'id': qid,
            **{name: ({'pred': data[qid].get('pred'), 'correct': bool(data[qid].get('correct', data[qid].get('judge')))} if qid in data else None) for name, data in all_lb.items()}})
    diagnostics = audited['diagnostics']
    pair_journals = set()
    for dataset in ('math', 'longbench'):
        rows = mapped(run/dataset/'predictions.jsonl')
        calls = [c for row in rows.values() for c in row['calls']]
        diagnostics[dataset].update({
            'repair_calls': sum('_repair' in c['tag'] for c in calls),
            'cache_hits': sum(c['cached'] for c in calls),
            'llm_calls': len(calls),
            'duplicate_executions': sum(len({json.dumps([e['name'], e['parameters']], sort_keys=True) for s in row['trace'] for e in s['executed']}) != sum(len(s['executed']) for s in row['trace']) for row in rows.values()),
            'sum_new_visible_characters': sum(s.get('progress', {}).get('new_visible_characters', 0) for row in rows.values() for s in row['trace']),
            'logical_prompt_tokens': sum(c['response'].get('usage', {}).get('prompt_tokens', 0) for c in calls),
            'logical_completion_tokens': sum(c['response'].get('usage', {}).get('completion_tokens', 0) for c in calls),
            'mean_final_prompt_tokens': mean(c['response']['usage']['prompt_tokens'] for c in calls if c['tag'] in ('math_repar_answer', 'v2_repar_answer')),
            'actual_search_channels': sum(len(e['observation'].get('query_channels', [])) for row in rows.values() for s in row['trace'] for e in s['executed']),
            'graph_neighbor_tasks': sum(any(e['name'] == 'graph_neighbors' for s in row['trace'] for e in s['executed']) for row in rows.values()),
            'paged_read_calls': sum(e['name'] == 'read' and 'offset' in e['parameters'] for row in rows.values() for s in row['trace'] for e in s['executed']),
            'base_search_cache_hits': sum(event['cached'] for row in rows.values() for step in row['trace'] for e in step['executed'] for event in e['observation'].get('query_execution', [])),
            'base_search_executions': sum(not event['cached'] for row in rows.values() for step in row['trace'] for e in step['executed'] for event in e['observation'].get('query_execution', [])),
            'actual_tree_search_limits': dict(Counter(e['parameters']['top_k'] for row in rows.values() for step in row['trace'] for e in step['executed'] if e['name'] == 'tree_search')),
            'resource_parameter_calls': sum(c['tag'].startswith('v4_parameters_') for c in calls),
        })
        encoder_stats = [row['state'].get('source_similarity') for row in rows.values() if row['state'].get('source_similarity')]
        if encoder_stats:
            diagnostics[dataset]['auxiliary_encoder'] = {'model': encoder_stats[0]['model'], 'revision': encoder_stats[0]['revision'],
                'device': 'cpu', 'scoring_seconds_sum': sum(v['scoring_seconds'] for v in encoder_stats),
                'encoder_load_seconds_sum': sum(v['load_seconds'] for v in encoder_stats),
                'texts_encoded': sum(v['texts_encoded'] for v in encoder_stats),
                'encoding_windows': sum(v['encoding_windows'] for v in encoder_stats),
                'long_inputs_encoded_in_full': sum(v['long_inputs_encoded_in_full'] for v in encoder_stats),
                'vector_cache_hits': sum(v['vector_cache_hits'] for v in encoder_stats)}
        pair_stats = [row['state'].get('source_reranking') for row in rows.values() if row['state'].get('source_reranking')]
        if pair_stats:
            diagnostics[dataset]['pair_reranking'] = {'model': pair_stats[0]['model'], 'revision': pair_stats[0]['revision'],
                'device': 'cpu', 'scoring_seconds_sum': sum(v['scoring_seconds'] for v in pair_stats),
                'model_load_seconds_sum': sum(v['load_seconds'] for v in pair_stats),
                'query_source_pairs': sum(v['query_source_pairs'] for v in pair_stats),
                'encoded_pair_windows': sum(v['pair_windows'] for v in pair_stats)}
            pair_journals.update(Path(v['raw_pair_model_journal']['path']) for v in pair_stats if v.get('raw_pair_model_journal'))
        source_indexes = [row['state'].get('semantic_source_index') for row in rows.values() if row['state'].get('semantic_source_index')]
        if source_indexes:
            diagnostics[dataset]['semantic_source_index'] = {
                'newly_constructed': sum(not v['reused'] for v in source_indexes),
                'reused': sum(v['reused'] for v in source_indexes),
                'construction_wall_seconds': sum(v['wall_seconds'] for v in source_indexes if not v['reused']),
                'cache_load_wall_seconds': sum(v['wall_seconds'] for v in source_indexes if v['reused']),
                'query_seconds': sum(v['query_seconds'] for v in source_indexes),
                'query_calls': sum(v['query_calls'] for v in source_indexes),
                'unique_source_cache_files': len({v['path'] for v in source_indexes}),
                'source_partitions_changed': False}
    offline_files = sorted((run/'offline_encoding').glob('worker_session_*.jsonl'))
    if offline_files:
        known_caches = {row['state']['semantic_source_index']['path']: row['state']['semantic_source_index']['sha256']
                         for row in mapped(run/'longbench/predictions.jsonl').values()}
        records = [record for path in offline_files for record in read_jsonl(path)]
        for record in records:
            source = record['source_cache']
            if known_caches.get(source['path']) != source['sha256'] or sha(source['path']) != source['sha256']:
                raise ValueError('Offline vector-cache journal disagrees with the actual run')
            if protocol['identity']['references'].get(record['index']) != record['index_sha256']:
                raise ValueError('Offline vectors came from outside the frozen index set')
        builds = [r for r in records if not r['source_cache']['reused']]
        diagnostics['longbench']['parallel_offline_source_encoding'] = {
            'source_cache_lookups': len(records), 'new_builds': len(builds),
            'construction_attempts_can_include_concurrent_duplicates': True,
            'worker_count': len({r['worker_pid'] for r in records}),
            'construction_wall_seconds_summed_across_workers': sum(r['source_cache']['wall_seconds'] for r in builds),
            'encoding_windows': sum(r['encoding']['encoding_windows'] for r in builds),
            'journals_sha256': {p.relative_to(run).as_posix(): sha(p) for p in offline_files}}
    orchestration_files = sorted(p for p in (run/'orchestration_source').rglob('*') if p.is_file())
    if orchestration_files:
        manifest = read_json(run/'orchestration_source/manifest.json')
        for name, expected in manifest['scripts_sha256'].items():
            if sha(run/'orchestration_source'/name) != expected:
                raise ValueError('An archived execution scheduler changed')
        if sha(run/'protocol.json') != manifest['protocol_sha256']:
            raise ValueError('The execution scheduler referenced another frozen condition')
        diagnostics['longbench']['scheduling_accounting'] = manifest
    parallel_files = sorted(p for p in (run/'parallel_execution').rglob('*') if p.is_file())
    if parallel_files:
        execution = read_json(run/'parallel_execution/completion.json')
        handoff = read_json(run/'parallel_execution/handoff.json')
        if not execution['complete'] or execution['coverage'] != 125:
            raise ValueError('Parallel orchestration is not complete')
        if execution['protocol_sha256'] != sha(run/'protocol.json') or handoff['protocol_sha256'] != execution['protocol_sha256']:
            raise ValueError('Parallel scheduler used another frozen condition')
        if sha(run/'parallel_execution/run_parallel_frozen_evaluation.py') != handoff['script_sha256']:
            raise ValueError('Parallel scheduler source changed')
        for name, expected in handoff['archived_partial_sha256'].items():
            if sha(name) != expected:
                raise ValueError('Archived partial execution changed')
        diagnostics['longbench']['parallel_execution'] = execution
        diagnostics['longbench']['parallel_handoff'] = handoff
    grades = all_math[new_name]
    diagnostics['math']['judge_calls'] = sum(len(g['judge_calls']) for g in grades.values())
    diagnostics['math']['third_grade_cases'] = [qid for qid, g in grades.items() if len(g['grades']) == 3]
    diagnostics['math']['needs_review_cases'] = [qid for qid, g in grades.items() if any(v.get('needs_review') for v in g['grades'])]
    now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    targets = {'math_threshold': max(comparisons[n]['math_score'] for n in target_methods),
        'longbench_correct_threshold': max(comparisons[n]['longbench_correct'] for n in target_methods)}
    targets['math_exceeds_both'] = comparisons[new_name]['math_score'] > targets['math_threshold']
    targets['longbench_exceeds_both'] = comparisons[new_name]['longbench_correct'] > targets['longbench_correct_threshold']
    targets['both_exceeded'] = targets['math_exceeds_both'] and targets['longbench_exceeds_both']
    analysis = {'completed_at': now, 'comparisons': comparisons, 'longbench_pairs': pairs, 'targets': targets,
        'diagnostics': diagnostics, 'source_sha256': source_hashes,
        'scope': f"All 19 MATH and 125 LongBench tasks with {solver['model']}. Historical comparisons reused.",
        'limitations': ['Known benchmark development iteration, not unseen confirmation',
            'Different controller/evidence budgets; no single-mechanism causal claim',
            'Shared hardware; cache or wall time does not establish inference acceleration',
            'Experimental math grading, anonymous27B judge; reference proofs not mandatory']}
    write_json(run/'analysis.json', analysis)
    write_json(run/'per_question.json', details)
    report_name = config['report_filename']
    lines = [f"# {new_name} 双数据集完整结果", '', f"实验冻结于 {protocol['frozen_at']}，Asia/Shanghai。汇总于 {now}。", '',
        f"{new_name} 使用 {solver['model']} 完整评测两个数据集。其余成绩引用历史条件。原教材及 126 份原索引按冻结哈希复用。", '',
        '| 方法 | MATH实验分 /150 | 整题全对 | LongBench正确 /125 | 准确率 |', '|---|---:|---:|---:|---:|']
    for name, row in comparisons.items():
        lb_count = str(row['longbench_correct']) if row['longbench_complete'] else f"未完成（{row['longbench_completed']}/125）"
        lb_accuracy = f"{row['longbench_correct']/125:.1%}" if row['longbench_complete'] else '—'
        lines.append(f"| {name} | {row['math_score']:.3f} | {row['math_fully_correct']}/19 | {lb_count} | {lb_accuracy} |")
    if protocol['identity']['solver'].get('source_similarity'):
        lines += ['', '本条件新增冻结的multilingual-e5-small CPU检索编码器，负责原文窗口与保留视图排序，不生成答案。Qwen作答底座不变，但辅助模型/计算条件不同，不能将分差全部归为纯工作流或等计算预算收益。编码器身份、完整长输入窗口、编码耗时和缓存分别记录。原有知识库索引未重建。']
    if pair_journals:
        lines += ['', f"原文候选与可见窗口使用冻结的mMARCO MiniLM CPU模型进行相关性排序。\n数学耗时合计{diagnostics['math']['pair_reranking']['scoring_seconds_sum']:.1f}秒，LongBench耗时合计{diagnostics['longbench']['pair_reranking']['scoring_seconds_sum']:.1f}秒。原始输入、logit、候选顺序和辅助权重均有审计记录。"]
    if any(not row['longbench_complete'] for row in comparisons.values()):
        lines += ['', '部分开发条件的完整MATH已未达到预先设定目标，且共享单槽位服务；为优先测试下一修订而停止其剩余LongBench。原始前缀、缓存和续跑状态保留；不将部分正确率当全量成绩，不对它们作配对比较。报告归档了对应原始前缀。']
    lines += ['', f"同时超过v2/v3的目标是否达到：{targets['both_exceeded']}；数学须高于{targets['math_threshold']}，LongBench须多于{targets['longbench_correct_threshold']}题。", '']
    lines += ['', '## 执行与完整性', '', f"全部19/125唯一覆盖，审计{audited['check_count']}项通过。原始预测、调用、来源、候选与选择、原文显示范围和计分均保留。", '',
        '| 指标 | MATH | LongBench |', '|---|---:|---:|']
    for key in ['no_evidence', 'mean_evidence_sources', 'control_errors', 'duplicate_executions', 'repair_calls', 'final_length', 'llm_calls', 'resource_parameter_calls', 'cache_hits', 'logical_prompt_tokens', 'logical_completion_tokens', 'mean_final_prompt_tokens', 'actual_search_channels', 'base_search_executions', 'base_search_cache_hits', 'graph_neighbor_tasks', 'paged_read_calls']:
        lines.append(f"| {key} | {diagnostics['math'][key]} | {diagnostics['longbench'][key]} |")
    if offline_files:
        offline = diagnostics['longbench']['parallel_offline_source_encoding']
        lines += ['', f"LongBench原文向量由{offline['worker_count']}个CPU进程并行预建。离线日志记录{offline['new_builds']}次冷建计算（含并发重复计算），进程计时之和为{offline['construction_wall_seconds_summed_across_workers']:.1f}秒。逐题在线记录另含冷建及缓存加载耗时。"]
    if orchestration_files:
        attempt = manifest['interrupted_encoding_attempt']
        lines += ['', f"第{attempt['after_completed_count']+1}题之前有一次中断的冷建尝试，估计耗时{attempt['approx_wall_seconds']:.1f}秒；该估计含调度/加载开销，未包含在保留预测的逐题计时中。调度程序和执行说明已归档。"]
    if parallel_files:
        lines += ['', f"LongBench 从第 {handoff['completed_prefix']} 条已保存预测接续，{execution['workers']} 个 CPU 工作进程共享 {execution['single_solver_service']}。父进程统一记录预测，并在生成后计分。调度记录、执行源码和传输日志见 parallel_execution。"]
    lines += ['', '## 配对与解释限制', '', f'LongBench {new_name}相对历史条件：', '']
    for name in target_methods:
        p = pairs[name]
        lines.append(f"- vs {name}：新独对{len(p['new_only_correct'])}题，旧独对{len(p['old_only_correct'])}题，双侧精确McNemar p={p['exact_mcnemar_p_unadjusted']:.4f}，未作多重校正。")
    if solver['model'] == config['judge']['model']:
        lines += ['', 'MATH 开放题的作答与匿名评审使用同一 27B 模型，判断可能存在相关偏差。']
    lines += ['', '这是已见数据上的开发实验。数学沿用匿名双评、分歧三评和等价解判分，报告实验折算分。LongBench 沿用原答案解析。', '',
        '最终生成仍为MATH4096-token自由文本、LongBench128-token原grammar。没有新增计算器、任务分类、最终答案恢复、复核或重答阶段。生成重复及长度终止如实保留，不按结果修订本条件。', '',
        '成本包含控制/修复请求，旧版存在缓存复用且预算不同，不据此宣称隔离的速度提升。完整逐题结果见per_question.json，机器统计见analysis.json，冻结条件见protocol.json，审计见audit.json。', '']
    (run/report_name).write_text('\n'.join(lines), encoding='utf-8')
    target = run/'postprocessing_source'/Path(__file__).name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, target)
    audit_source = run/'postprocessing_source/audit_repar_v4.py'
    shutil.copyfile(Path(__file__).with_name('audit_repar_v4.py'), audit_source)
    completion = {'complete': True, 'completed_at': now, 'coverage': {'math': 19, 'longbench': 125},
        'audit_checks': audited['check_count'], 'frozen_generation': frozen_summary,
        'files_sha256': {p: sha(run/p) for p in ['math/predictions.jsonl', 'math/grades.jsonl', 'longbench/predictions.jsonl', 'audit.json', 'analysis.json', 'per_question.json', report_name,
            *[p.relative_to(run).as_posix() for p in offline_files],
            *[p.relative_to(run).as_posix() for p in sorted(pair_journals)],
            *[p.relative_to(run).as_posix() for p in orchestration_files],
            *[p.relative_to(run).as_posix() for p in parallel_files],
            *[p.relative_to(run).as_posix() for p in (run/'longbench/scheduler_handoff.json', run/'longbench/ready_schedule.jsonl') if p.exists()],
            *[p.relative_to(run).as_posix() for p in sorted((run/'development_prefixes').rglob('predictions.jsonl'))]]},
        'postprocessing_source_sha256': sha(target),
        'postprocessing_source_files_sha256': {target.name: sha(target), audit_source.name: sha(audit_source)}}
    write_json(run/'completion.json', completion)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='artifacts/repar_v4_20261119')
    finalize(parser.parse_args().run)
