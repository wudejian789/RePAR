"""Audit the paired r27/r28 MATH runs and publish the r28 result table."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
from statistics import mean

from repar.benchmarks.math_exam import aggregate_grades, stitch_context
from repar.benchmarks.math_judge import judge_messages, judge_schema, validate_grade
from repar.benchmarks.math_suite import usage
from repar.io import digest, read_json, write_json
from repar.v4.workflow import LITERAL_PLAN_PROMPT, LITERAL_PLAN_SCHEMA, bind_query

from grade_math_08_temperature import questions_and_judge
from regrade_math_2025 import CHOICE_BUDGET, CHOICE_TYPES, choice_input, choice_score
from repeat_benchmark_recommended import configuration as recommended_configuration
from report_benchmark_recommended import audit_condition
from report_math_08_temperature import saved_request
from report_math_2025_regrade import audit_requests, score_judged_choice
from resume_math_2025_regrade import choice_observation
from run_benchmark_recommended import OLD, sha
from run_math_r28 import CONFIG, MODELS, SEEDS, configuration


REFERENCE_08 = Path('artifacts/math_08_temperature_01_20261124')
REFERENCE_27 = Path('artifacts/math_2025_national_i_27b_judge_20261124')
START = '## RePAR v4-r28 MATH 查询原文约束'
HAN = re.compile(r'[\u3400-\u9fff]')
LATIN = re.compile('[A-Za-z]')


def tree_query_counts(trace):
    searches = [item['parameters'] for step in trace for item in step['executed'] if item['name'] == 'tree_search']
    no_han = [item for item in searches if LATIN.search(item['query']) and not HAN.search(item['query'])]
    return {'tree_searches': len(searches), 'latin_no_han': len(no_han),
            'latin_no_han_without_anchor': sum('anchor_query' not in item for item in no_han)}


def audit_bindings(folder, row, context, query_task):
    plans = {}
    for call in row['calls']:
        if call['tag'].startswith('v4_plan_'):
            step = int(call['tag'].split('_')[2])
            plans[step] = read_json(folder / 'cache/solver' / f"{call['key']}.json")
    exposed = {event['stage']: event['spans'] for event in row['state']['exposures']}
    stats = Counter()
    for step in row['trace']:
        if 'query_binding' not in step:
            assert not step['executed'] and 'error' in step
            continue
        number = step['round']
        call = plans[number]
        body = call['request']
        assert body['messages'][0]['content'] == LITERAL_PLAN_PROMPT
        assert body['response_format']['json_schema']['schema'] == LITERAL_PLAN_SCHEMA
        state = json.loads(body['messages'][1]['content'])
        assert state['task'] == query_task
        spans = exposed.get(f'controller_{number}', [])
        sources = [context[span['start']:span['end']] for span in spans]
        assert all(text in state['source_evidence'] for text in sources)
        control = json.loads(call['response']['choices'][0]['message']['content'])
        query, binding = bind_query(control['query_phrases'], query_task, sources)
        assert step['query_binding'] == binding
        assert step['plan'] == {'query': query, 'objective': control['objective']}
        stats['plans'] += 1
        stats['fallback'] += binding['used_original_task']
        stats['rejected_phrases'] += len(binding['rejected'])
        for execution in step['executed']:
            params = execution['parameters']
            if 'query' in params:
                assert params['query'] == query
                assert params.get('anchor_query', query_task) == query_task
                stats['queries'] += 1
    return stats


def audit_grade(cache, grade, question, answer, judge):
    assert grade['id'] == question['id'] and grade['response_sha256'] == digest(answer)
    if question['question_type'] in CHOICE_TYPES:
        messages, schema = choice_input(question, answer)
        raws = audit_requests(cache, grade['judge_calls'], judge, messages, schema, CHOICE_BUDGET,
                              len(grade['observations']))
        observed = [choice_observation(question, answer, raw) for raw in raws]
        assert observed == grade['observations']
        vote = choice_score(question, observed)
        expected = {**vote, **score_judged_choice(question, vote['prediction'])}
        assert grade['prediction'] == expected['prediction']
    else:
        audit_requests(cache, grade['judge_calls'], judge, judge_messages(question, answer), judge_schema(question),
                       judge['max_tokens'], len(grade['grades']))
        expected = aggregate_grades(question, [validate_grade(question, value) for value in grade['grades']])
        if question['question_type'] == 'fill_in_blank':
            assert grade['score'] in {0, question['score']}
    assert all(grade[key] == expected[key] for key in ('score', 'fully_correct', 'review_flag'))


def audit_run(settings, model, questions, judge, deployment, context, previous):
    root = Path(settings['output'])
    folder = root / model / 'math/repar'
    seed = settings['answer_seed']
    assert {p.stem for p in (folder / 'cases').glob('*.json')} == questions.keys()
    assert {p.stem for p in (folder / 'grades').glob('*.json')} == questions.keys()
    manifest = read_json(root / f'{model}_judge_protocol.json')
    assert manifest['deployment'] == deployment and manifest['config']['temperature'] == 0
    for path, expected in manifest['files'].items():
        assert sha(Path(path)) == expected, path

    prior_root = (REFERENCE_08 / f'seed_{seed}' if model == '08'
                  else Path(recommended_configuration(seed)['output']))
    prior_folder = prior_root / model / 'math/repar'
    grades, changed, controls, judge_calls = {}, {}, Counter(), []
    previous_queries, current_queries = Counter(), Counter()
    for qid, question in questions.items():
        row = read_json(folder / 'cases' / f'{qid}.json')
        old = read_json(prior_folder / 'cases' / f'{qid}.json')
        assert row['status'] == 'ok' and row['agent_version'] == settings['agent_version']
        assert row['query_tokenizer'] == 'chinese' and row['context_sha256'] == old['context_sha256'] == digest(context)
        assert row['task'] == old['task'] and row['index_reuse']['sha256'] == old['index_reuse']['sha256']
        body = saved_request(folder, row)
        assert body == dict(saved_request(prior_folder, old), messages=body['messages'])
        task = json.loads(row['task'])
        query_task = '\n'.join([task['question_stem'], *(item['text'] for item in task['choices']),
                                *(item['question'] for item in task['subquestions'])])
        controls.update(audit_bindings(folder, row, context, query_task))
        previous_queries.update(tree_query_counts(old['trace']))
        current_queries.update(tree_query_counts(row['trace']))
        grade = read_json(folder / 'grades' / f'{qid}.json')
        audit_grade(folder / 'cache/judge', grade, question, row['response'], judge)
        judge_calls.extend(grade['judge_calls'])
        grades[qid] = grade['score']
        changed[qid] = {
            'r27_score': previous[qid]['score'], 'r28_score': grade['score'],
            'r27_finish': old['calls'][-1]['response']['choices'][0]['finish_reason'],
            'r28_finish': row['calls'][-1]['response']['choices'][0]['finish_reason'],
            'termination': row['termination'], 'fully_correct': grade['fully_correct'],
        }
    result = audit_condition(settings, root, model, 'math', 'repar', questions, judge, False)
    result.update(score=sum(grades.values()), grade_count=len(grades), complete=True,
                  full=sum(item['fully_correct'] for item in changed.values()),
                  judge_usage=usage(judge_calls), controls=dict(controls), comparisons=changed,
                  tree_queries={'r27': dict(previous_queries), 'r28': dict(current_queries)})
    return result


def main():
    questions, judge, deployment = questions_and_judge()
    source = read_json(OLD['08'] / 'protocol.json')['identity']['config']
    context, _ = stitch_context(source['math_textbook'])
    root = Path(read_json(CONFIG)['output'])
    previous = {model: read_json(path / 'summary.json')
                for model, path in zip(MODELS, (REFERENCE_08, REFERENCE_27))}
    runs = {}
    for seed in SEEDS:
        settings = configuration(seed)
        seed_root = Path(settings['output'])
        frozen = read_json(seed_root / 'protocol.json')['identity']
        assert frozen['configuration'] == settings
        for identities in (frozen['files'], frozen['core'], read_json(seed_root / 'r28_implementation.json')):
            for path, expected in identities.items():
                assert sha(Path(path)) == expected, path
        runs[seed] = {}
        for model in MODELS:
            key = f'{seed}/repar' if model == '08' else f'{seed}/{model}/repar'
            grade = previous[model]['math_grades'][key]
            if model == '27':
                grade = grade['grades']
            runs[seed][model] = audit_run(settings, model, questions, judge, deployment, context, grade)

    sections = [
        START, '',
        '两种基模各用种子 42、43、44 完整重跑 19 题，共审计 114 份作答及其 27B 匿名评分。'
        '使用同一中文索引、自由文本和 4,096-token 上限。0.8B 温度为 0.1，27B 为 0.7。'
        '作答耗时包括在线控制、检索和请求等待。两版共用已构建的中文索引。', '',
        '| 基模 | r27 三轮均分 /150 | r28 三轮分数 /150 | r28 均分 /150 | 平均分差 | r27 / r28 触顶数 | r28 秒/题 | r28 输入 / 输出 token 每轮均值 |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    means = {}
    for model, name in zip(MODELS, ('0.8B', '27B')):
        rows = [runs[seed][model] for seed in SEEDS]
        old = [previous[model]['runs'][str(seed)][f'{model}/math']['repar'] for seed in SEEDS]
        cap_transitions = {}
        for row in rows:
            for pair in row['comparisons'].values():
                transition = f"{pair['r27_finish']}_to_{pair['r28_finish']}"
                group = cap_transitions.setdefault(transition, {'count': 0, 'score_delta': 0})
                group['count'] += 1
                group['score_delta'] += pair['r28_score'] - pair['r27_score']
        queries = {}
        for version in ('r27', 'r28'):
            totals = Counter()
            for row in rows:
                totals.update(row['tree_queries'][version])
            queries[version] = dict(totals)
        stats = {'r27_mean': mean(r['score'] for r in old), 'r28_mean': mean(r['score'] for r in rows),
                 'r28_scores': [r['score'] for r in rows],
                 'paired_deltas': [r['score'] - o['score'] for r, o in zip(rows, old)],
                 'r27_cap': sum(r['answer_finish'].get('length', 0) for r in old),
                 'r28_cap': sum(r['answer_finish'].get('length', 0) for r in rows),
                 'cap_transitions': cap_transitions,
                 'tree_queries': queries,
                 'r27_seconds_per_case': mean(r['case_mean'] for r in old),
                 'r28_seconds_per_case': mean(r['case_mean'] for r in rows),
                 'mean_input_tokens': mean(r['costs']['input']['sum'] for r in rows),
                 'mean_output_tokens': mean(r['costs']['output']['sum'] for r in rows)}
        means[model] = stats
        scores = '、'.join(f'{score:.3f}' for score in stats['r28_scores'])
        sections.append(f"| {name} | {stats['r27_mean']:.3f} | {scores} | {stats['r28_mean']:.3f} | "
                        f"{mean(stats['paired_deltas']):+.3f} | {stats['r27_cap']} / {stats['r28_cap']} | "
                        f"{stats['r28_seconds_per_case']:.2f} | "
                        f"{stats['mean_input_tokens']:,.0f} / {stats['mean_output_tokens']:,.0f} |")
    sections.extend(['', f"[逐题分差、查询审计及耗时明细]({root.as_posix()}/summary.json)"])
    write_json(root / 'summary.json', {'updated_at': datetime.now().astimezone().isoformat(),
                                     'graded_count': 114, 'means': means, 'runs': runs})
    report = Path('all_results_new.md')
    text = report.read_text(encoding='utf-8').split(START, 1)[0].rstrip()
    temporary = report.with_suffix('.md.tmp')
    temporary.write_text(text + '\n\n' + '\n'.join(sections) + '\n', encoding='utf-8')
    temporary.replace(report)
    print(json.dumps(means, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
