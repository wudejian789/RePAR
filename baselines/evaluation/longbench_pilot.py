"""Length-only, cold-cache feasibility screen. Not a full accuracy benchmark."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import requests
from repar.io import read_json, write_json
from repar.model_support import BoundedLLMClient, SentenceEncoder
from repar.evaluation import extract_answer, qa_text
from baselines._factory import create_retriever
from baselines._pipeline import retrieve_context
from .longbench import prompt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call_stats(calls):
    usage = [x['response'].get('usage', {}) for x in calls]
    return {'calls': len(calls), 'cache_hits': sum(x['cached'] for x in calls),
        'prompt_tokens': sum(x.get('prompt_tokens', 0) for x in usage),
        'completion_tokens': sum(x.get('completion_tokens', 0) for x in usage),
        'length_finishes': sum(x['response']['choices'][0].get('finish_reason') == 'length' for x in calls),
        'seconds': sum(x.get('seconds', 0) for x in calls)}


def worker(protocol_path, method, sample_id):
    protocol = read_json(protocol_path)
    config = protocol['model_config']
    item = next(x for x in read_json(protocol['dataset']) if x['_id'] == sample_id)
    spec = next(x for x in protocol['samples'] if x['id'] == sample_id)
    if hashlib.sha256(item['context'].encode()).hexdigest() != spec['context_sha256']:
        raise ValueError('Dataset context changed since protocol freeze')
    run = Path(protocol_path).parent / method / sample_id
    run.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = time.monotonic() + protocol['case_seconds'] - 5
    client = BoundedLLMClient(config, run / 'cache', deadline=deadline, call_seconds=protocol['call_seconds'], journal=run / 'calls.jsonl')
    final = BoundedLLMClient(config, run / 'cache', deadline=deadline, call_seconds=protocol['call_seconds'], journal=run / 'calls.jsonl', answer_grammar=True)
    source = None
    result = {'method': method, 'id': sample_id, 'length': spec['length'], 'words': spec['words'], 'status': 'started'}
    def save():
        result['worker_seconds'] = time.perf_counter()-started
        result['index_llm'] = call_stats(client.calls)
        result['answer_llm'] = call_stats(final.calls)
        if source is not None:
            result['index_stats'] = source.stats
        write_json(run / 'result.json', result)
    try:
        t = time.perf_counter()
        encoder = None if method == 'BM25_RAG' else SentenceEncoder(protocol['encoder'])
        result['encoder_load_seconds'] = time.perf_counter()-t
        t = time.perf_counter()
        result['phase'] = 'build'
        save()
        source = create_retriever(method, item['context'], client, encoder, working_dir=run / 'index',
            chunk_words=config['chunk_words'], parameters=protocol['method_parameters'].get(method, {}))
        result['build_seconds'] = time.perf_counter()-t
        result['build_llm'] = call_stats(client.calls)
        result['phase'] = 'retrieval'
        save()
        t = time.perf_counter()
        context, records = retrieve_context(qa_text(item), source, config['retrieval_top_k'])
        result['retrieval_seconds'] = time.perf_counter()-t
        result['retrieval_llm_calls'] = len(client.calls) - result['build_llm']['calls']
        result['evidence'] = records
        if encoder:
            result['embedding'] = {'calls': len(encoder.calls), 'seconds': sum(x['seconds'] for x in encoder.calls),
                'texts': sum(x['texts'] for x in encoder.calls), 'windows': sum(x['windows'] for x in encoder.calls)}
        if not records:
            raise RuntimeError('No evidence retrieved')
        result['phase'] = 'answer'
        save()
        t = time.perf_counter()
        response = final.complete([{'role': 'user', 'content': prompt(item, context)}], config['answer_max_tokens'], tag='baseline_longbench_answer')
        result['answer_seconds'] = time.perf_counter()-t
        result.update(response=response, pred=extract_answer(response), gold=item['answer'])
        result['correct'] = result['pred'] == item['answer']
        result['status'] = 'completed' if result['pred'] else 'answer_parse_error'
        result['phase'] = 'done'
    except Exception as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        save()
        if source is not None and hasattr(source, 'close'):
            source.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baseline_pilot.json')
    parser.add_argument('--output')
    parser.add_argument('--worker', nargs=3, metavar=('PROTOCOL','METHOD','ID'))
    args = parser.parse_args()
    if args.worker:
        result = worker(*args.worker)
        print(json.dumps({k: result[k] for k in ('method','id','status','worker_seconds')}, ensure_ascii=False), flush=True)
        return
    settings = read_json(args.config)
    output = Path(args.output or ('artifacts/baseline_pilot_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = Path(settings['dataset']).resolve()
    items = [x for x in read_json(dataset) if x['domain'] == 'Multi-Document QA']
    samples = []
    for length in ('short', 'medium', 'long'):
        item = min((x for x in items if x['length'] == length), key=lambda x: (len(x['context'].split()), x['_id']))
        samples.append({'id': item['_id'], 'length': length, 'words': len(item['context'].split()),
            'context_sha256': hashlib.sha256(item['context'].encode()).hexdigest()})
    root = Path(__file__).resolve().parents[2]
    sources = list((root / 'baselines').rglob('*.py')) + list((root / 'src/repar').rglob('*.py'))
    source_hashes = {p.relative_to(root).as_posix(): sha(p) for p in sources}
    protocol = {**settings, 'dataset': str(dataset), 'dataset_sha256': sha(dataset),
        'model_config': read_json(settings['model_config']), 'encoder': str(Path(settings['encoder']).resolve()),
        'samples': samples, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_hashes': source_hashes, 'selection': 'Shortest full context in each length stratum, independent of labels; optimistic feasibility screen only',
        'gate': 'After the first non-completed case in a method, longer cases are not run. No accuracy-based gate.',
        'cache_policy': 'New per-case cache and index, no historical request reuse; service prompt-cache state is not reset',
        'limitations': 'Three lower-length-bound cases cannot estimate accuracy or mean/full-suite runtime. Same machine serial runs; no model training.'}
    for method in settings['methods']:
        source_dir = root/'baselines'/method
        if (source_dir/'sources.json').exists():
            write_json(output / 'source_manifests' / (method+'.json'), read_json(source_dir/'sources.json'))
    for relative in source_hashes:
        dest = output/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root/relative).read_bytes())
    write_json(output/'protocol.json', protocol)
    print(f'FROZEN {output}', flush=True)
    rows = []
    for method in settings['methods']:
        failed = False
        for sample in samples:
            case = output/method/sample['id']
            case.mkdir(parents=True, exist_ok=True)
            if failed:
                rows.append({'method': method, **sample, 'status': 'not_run_after_failed_gate'})
                continue
            print(f'START {method} {sample["length"]} {sample["words"]} words', flush=True)
            start = time.perf_counter()
            with (case/'stdout.log').open('w',encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-X','utf8','-m','baselines.evaluation.longbench_pilot',
                    '--worker',str(output/'protocol.json'),method,sample['id']], cwd=root, stdout=log, stderr=subprocess.STDOUT)
                try:
                    process.wait(timeout=settings['case_seconds'])
                    result = read_json(case/'result.json') if (case/'result.json').exists() else {'status':'worker_crash','returncode':process.returncode}
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    result = read_json(case/'result.json') if (case/'result.json').exists() else {}
                    result.update(status='hard_timeout', limit_seconds=settings['case_seconds'])
                    write_json(case/'result.json', result)
            result.update(method=method, id=sample['id'], length=sample['length'], words=sample['words'], end_to_end_seconds=time.perf_counter()-start)
            # A killed worker may not have checkpointed the last calls; journal stays authoritative.
            journal = case/'calls.jsonl'
            events = [json.loads(x) for x in journal.read_text(encoding='utf-8').splitlines()] if journal.exists() else []
            result['journal_completions'] = sum(x.get('event')=='completion' for x in events)
            result['journal_http_errors'] = sum(x.get('event')=='http_error' for x in events)
            result['journal_prompt_tokens'] = sum(x.get('response',{}).get('usage',{}).get('prompt_tokens',0) for x in events)
            result['journal_completion_tokens'] = sum(x.get('response',{}).get('usage',{}).get('completion_tokens',0) for x in events)
            write_json(case/'result.json', result)
            rows.append({k:v for k,v in result.items() if k not in ('evidence','traceback')})
            write_json(output/'summary.json', rows)
            print(f'END {method} {sample["length"]} {result["status"]} {result["end_to_end_seconds"]:.1f}s', flush=True)
            failed = result['status'] != 'completed'
            health = requests.get(protocol['model_config']['base_url']+'/health', timeout=10)
            health.raise_for_status()
    write_json(output/'summary.json', rows)
    print(f'FINISHED {output}', flush=True)


if __name__ == '__main__':
    main()
