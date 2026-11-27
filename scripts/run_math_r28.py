"""Rerun both MATH solvers with the source-language query controller."""
from argparse import ArgumentParser
from pathlib import Path
import time

import requests

from repar.io import digest, read_json, write_json
from repar.v4 import selection, workflow

from grade_math_08_temperature import mark, questions_and_judge
from recover_math_2025_judge_quotes import QuoteRepairClient
import regrade_math_2025 as grading
from repeat_benchmark_recommended import SEEDS, configuration as recommended_configuration
import resume_benchmark_recommended as resume
from resume_math_2025_regrade import choice_observation
import run_benchmark_recommended as runner


CONFIG = Path('configs/math_r28_20261124.json')
MODELS = ('08', '27')


def configuration(seed):
    experiment = read_json(CONFIG)
    temperature = read_json(experiment['temperature_reference'])['temperature']
    settings = recommended_configuration(seed)
    settings.update(output=f"{experiment['output']}/seed_{seed}", experiment_id=f"{experiment['experiment_id']}/seed_{seed}",
                    conditions={f'{model}/math': ['repar'] for model in MODELS}, longbench_08_parallel_endpoints=[])
    settings['sampling']['08']['temperature'] = temperature
    settings['agent_version'] = experiment['agent_version']
    return settings


def prepare_generation(seed):
    settings = configuration(seed)
    protocols = {name: read_json(path / 'protocol.json') for name, path in runner.OLD.items()}
    for protocol in protocols.values():
        protocol['identity']['solver'].update(seed=seed, agent_version=settings['agent_version'])
    config = protocols['08']['identity']['config']
    runner.preflight(settings, protocols, config)
    files = (CONFIG, Path(__file__), Path(resume.__file__), Path('scripts/recover_benchmark_recommended.py'),
             Path('scripts/repeat_benchmark_recommended.py'), Path(read_json(CONFIG)['temperature_reference']),
             runner.CHINESE / 'zh_index.py')
    identity = {p.as_posix(): runner.sha(p) for p in files}
    manifest = Path(settings['output']) / 'r28_implementation.json'
    if manifest.exists():
        assert read_json(manifest) == identity
    else:
        write_json(manifest, identity)
    return settings, protocols, config


def generate(model):
    runner.prepare = resume.prepare
    runner.case = resume.one_case
    workflow.structured = selection.structured = resume.structured_unicode
    for seed in SEEDS:
        settings, protocols, config = prepare_generation(seed)
        runner.run(settings, protocols, config, select=[f'{model}/math'])


def grade(model, watch):
    questions, judge, expected = questions_and_judge()
    response = requests.get(judge['base_url'] + '/props', timeout=15)
    response.raise_for_status()
    actual = response.json()
    for key in ('model_alias', 'model_path', 'build_info', 'total_slots', 'default_generation_settings', 'chat_template'):
        assert actual[key] == expected[key], key
    grading.choice_observation = choice_observation
    for seed in SEEDS:
        settings = configuration(seed)
        root = Path(settings['output'])
        folder = root / model / 'math/repar'
        config = {**judge, 'deployment_id': f"{judge['deployment_id']}/{settings['experiment_id']}"}
        clients = [QuoteRepairClient(dict(config, seed=s), folder / 'cache/judge', call_seconds=180)
                   for s in config['replicate_seeds']]
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / f'{model}_judge_protocol.json'
        identity = {'config': config, 'deployment': expected,
                    'files': {str(p): runner.sha(p) for p in (CONFIG, Path(__file__), Path('scripts/grade_math_08_temperature.py'),
                              Path('scripts/regrade_math_2025.py'), Path('scripts/resume_math_2025_regrade.py'),
                              Path('scripts/recover_math_2025_judge_quotes.py'), grading.ROOT / 'protocol.json')}}
        if manifest.exists():
            assert read_json(manifest) == identity
        else:
            write_json(manifest, identity)
        for qid, question in questions.items():
            dest = folder / 'grades' / f'{qid}.json'
            if dest.exists():
                continue
            prediction = folder / 'cases' / f'{qid}.json'
            while watch and not prediction.exists():
                time.sleep(5)
            if not prediction.exists():
                raise FileNotFoundError(prediction)
            record = read_json(prediction)
            assert record['id'] == qid and record['status'] == 'ok' and record['agent_version'] == settings['agent_version']
            answer = record['response']
            result = mark(question, answer, clients, judge)
            write_json(dest, {**result, 'id': qid, 'response_sha256': digest(answer)})
            print('GRADED', seed, model, qid, result['score'], flush=True)


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('freeze', 'generate', 'grade'))
    cli.add_argument('--model', choices=MODELS)
    cli.add_argument('--watch', action='store_true')
    args = cli.parse_args()
    if args.phase == 'freeze':
        for seed in SEEDS:
            prepare_generation(seed)
    elif args.model is None:
        cli.error('--model is required for generation and grading')
    elif args.phase == 'generate':
        generate(args.model)
    else:
        grade(args.model, args.watch)
