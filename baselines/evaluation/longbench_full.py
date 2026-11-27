"""Frozen, resumable full-condition evaluation; failures never remove dataset IDs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback

import requests

from repar.concurrent_client import ConcurrentLLMClient
from repar.evaluation import extract_answer, qa_text
from repar.model_support import SentenceEncoder
from baselines._factory import create_retriever
from baselines._pipeline import retrieve_context
from .longbench import prompt
from .longbench_pilot import call_stats


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary,path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_sources(protocol, snapshot):
    for relative, expected in protocol['source_hashes'].items():
        if sha(Path(snapshot)/relative) != expected:
            raise RuntimeError(f'Frozen executable source changed: {relative}')


def preflight(settings):
    metadata = {}
    weight_hashes = {}
    for endpoint in settings['endpoints']:
        requests.get(endpoint+'/health',timeout=10).raise_for_status()
        data = requests.get(endpoint+'/props',timeout=10).json()
        if data['model_alias'] != 'Qwen3.5-0.8B-Q4_K_M':
            raise RuntimeError('Wrong serving model')
        if data['total_slots'] != settings['endpoint_slots']:
            raise RuntimeError('Serving slot count differs from protocol')
        if data['default_generation_settings']['n_ctx'] != settings['context_per_slot']:
            raise RuntimeError('Serving context per slot differs from protocol')
        if data['default_generation_settings']['params']['speculative.types'] != 'none':
            raise RuntimeError('Unexpected speculative decoding')
        path = data['model_path']
        if path not in weight_hashes:
            weight_hashes[path] = sha(path)
        metadata[endpoint] = {'props':data,'weight_sha256':weight_hashes[path]}
    if len(set(weight_hashes.values())) != 1:
        raise RuntimeError('Replica weights do not match')
    return metadata


def prepare(config_path, output):
    settings = read(config_path)
    services = preflight(settings)
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    items = [x for x in read(ROOT/settings['dataset']) if x['domain']==settings['domain']]
    if len(items)!=settings['expected_items'] or len({x['_id'] for x in items})!=len(items):
        raise RuntimeError('Dataset coverage/unique IDs do not match the declared condition')
    write(output/'input/items.json',items)
    snapshot = output/'source'
    files = []
    for folder in ('baselines','src/repar'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and
                     (p.suffix in ('.py','.json','.txt') or p.name=='README.md'))
    files.extend(ROOT/p for p in ('scripts/run_baselines_full.py','scripts/serve_local.ps1',
                                 'scripts/probe_baseline_throughput.py','pyproject.toml'))
    source_hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        dest = snapshot/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        source_hashes[relative.as_posix()] = sha(dest)
    model = read(ROOT/settings['model_config'])
    model.update(base_url=f'http://127.0.0.1:{settings["gateway_port"]}',
                 deployment_id=settings['deployment_id'],context_window=settings['context_per_slot'])
    parameters = settings['method_parameters']
    for method in ('HippoRAG2','HyperGraphRAG'):
        parameters[method]['workers'] = settings['index_workers']
    protocol = {**settings,'model_config':model,'method_parameters':parameters,
        'repository':str(ROOT),'snapshot':str(snapshot),'run':str(output),
        'dataset':str(output/'input/items.json'),'dataset_sha256':sha(ROOT/settings['dataset']),
        'frozen_items_sha256':sha(output/'input/items.json'),
        'encoder':str((ROOT/settings['encoder']).resolve()),'source_hashes':source_hashes,
        'serving':services,'frozen_at_utc':utc(),'python':sys.version,'platform':platform.platform(),
        'items':[{'id':x['_id'],'length':x['length'],'words':len(x['context'].split()),
                  'context_sha256':hashlib.sha256(x['context'].encode()).hexdigest()} for x in items],
        'scoring':'All 125 IDs form the denominator; terminal failures count as incorrect and are reported separately.',
        'cache_policy':'New full-run caches; no pilot/historical cache import. Resume reuses only exact request identities.',
        'execution':'Three method streams, independent chunk extraction within methods, shared global replica slot gateway.',
        'timing_caveat':'Parallel shared-GPU run, including another pre-existing math task; not isolated single-task latency.',
        'seen_data_disclosure':'Three length-minimum pilot questions were previously observed; no label-based tuning.'}
    write(output/'protocol.json',protocol)
    write(output/'progress.json',{'status':'prepared','expected':len(items)*len(settings['methods']),
                                  'terminal':0,'run':str(output)})
    return output/'protocol.json'


def case_worker(protocol_path, method, item_id, attempt_dir):
    protocol = read(protocol_path)
    verify_sources(protocol,protocol['snapshot'])
    item = next(x for x in read(protocol['dataset']) if x['_id']==item_id)
    spec = next(x for x in protocol['items'] if x['id']==item_id)
    assert hashlib.sha256(item['context'].encode()).hexdigest()==spec['context_sha256']
    attempt = Path(attempt_dir)
    case = attempt.parent.parent
    cfg = protocol['model_config']
    client = ConcurrentLLMClient(cfg,case/'cache',journal=attempt/'calls.jsonl',
        call_seconds=protocol['call_seconds'],case_id=f'{method}/{item_id}')
    final = ConcurrentLLMClient(cfg,case/'cache',journal=attempt/'calls.jsonl',
        call_seconds=protocol['call_seconds'],case_id=f'{method}/{item_id}',answer_grammar=True)
    result = {'method':method,'id':item_id,'length':item['length'],'difficulty':item.get('difficulty'),
        'domain':item['domain'],'context_sha256':spec['context_sha256'],'words':spec['words'],
        'question':qa_text(item),'status':'running','phase':'encoder','started_at_utc':utc()}
    source = encoder = None
    started = time.perf_counter()
    def save():
        result['worker_seconds'] = time.perf_counter()-started
        result['index_llm'] = call_stats(client.calls)
        result['answer_llm'] = call_stats(final.calls)
        result['transport_errors'] = client.errors+final.errors
        if source is not None:
            result['index_stats'] = source.stats
        if encoder is not None:
            result['embedding'] = {'calls':len(encoder.calls),
                'seconds':sum(x['seconds'] for x in encoder.calls),
                'texts':sum(x['texts'] for x in encoder.calls),
                'windows':sum(x['windows'] for x in encoder.calls)}
        write(attempt/'result.json',result)
    try:
        save()
        t = time.perf_counter()
        encoder = SentenceEncoder(protocol['encoder'],threads=protocol['encoder_threads'])
        result['encoder_load_seconds'] = time.perf_counter()-t
        result['phase'] = 'build'
        save()
        t = time.perf_counter()
        source = create_retriever(method,item['context'],client,encoder,working_dir=attempt/'index',
            chunk_words=cfg['chunk_words'],parameters=protocol['method_parameters'][method])
        result['build_seconds'] = time.perf_counter()-t
        result['build_llm'] = call_stats(client.calls)
        result['phase'] = 'retrieval'
        save()
        t = time.perf_counter()
        context,records = retrieve_context(qa_text(item),source,cfg['retrieval_top_k'])
        result['retrieval_seconds'] = time.perf_counter()-t
        result['retrieval_llm_calls'] = len(client.calls)-result['build_llm']['calls']
        result['evidence'] = records
        if not records:
            raise RuntimeError('No evidence returned; no silent dense/BM25 substitution')
        result['phase'] = 'answer'
        save()
        t = time.perf_counter()
        response = final.complete([{'role':'user','content':prompt(item,context)}],cfg['answer_max_tokens'],
                                  tag='baseline_longbench_answer')
        result['answer_seconds'] = time.perf_counter()-t
        result.update(response=response,pred=extract_answer(response),gold=item['answer'])
        result['correct'] = result['pred']==item['answer']
        result['status'] = 'completed' if result['pred'] else 'answer_parse_error'
        result['phase'] = 'done'
    except Exception as exc:
        result.update(status='error',correct=False,error_type=type(exc).__name__,error=str(exc),
                      traceback=traceback.format_exc(),gold=item['answer'],
                      failed_phase_seconds=time.perf_counter()-t)
    finally:
        result['finished_at_utc'] = utc()
        save()
        if source is not None and hasattr(source,'close'):
            source.close()
    return result


def summarize(protocol, final=False):
    run = Path(protocol['run'])
    summary, predictions = {}, []
    expected_ids = {x['id'] for x in protocol['items']}
    for method in protocol['methods']:
        records = [read(p) for p in (run/method).glob('*/final.json')]
        ids = [x['id'] for x in records]
        assert len(ids)==len(set(ids)) and set(ids)<=expected_ids
        if final and set(ids)!=expected_ids:
            raise RuntimeError(f'Incomplete full condition: {method}')
        successes = [x for x in records if x['status']=='completed']
        row = {'expected':len(expected_ids),'terminal':len(records),'valid_answers':len(successes),
            'errors':len(records)-len(successes),'correct':sum(x.get('correct',False) for x in records),
            'accuracy_all_ids':sum(x.get('correct',False) for x in records)/len(expected_ids) if final else None,
            'sum_case_wall_seconds':sum(x.get('end_to_end_seconds',0) for x in records),
            'sum_build_seconds_completed':sum(x.get('build_seconds',0) for x in records),
            'llm_calls':sum(x.get('index_llm',{}).get('calls',0)+x.get('answer_llm',{}).get('calls',0) for x in records),
            'cache_hits':sum(x.get('index_llm',{}).get('cache_hits',0)+x.get('answer_llm',{}).get('cache_hits',0) for x in records),
            'length_finishes':sum(x.get('index_llm',{}).get('length_finishes',0)+x.get('answer_llm',{}).get('length_finishes',0) for x in records)}
        summary[method] = row
        predictions.extend(records)
    result = {'status':'complete' if final else 'running','updated_at_utc':utc(),
        'expected':len(expected_ids)*len(protocol['methods']),
        'terminal':sum(x['terminal'] for x in summary.values()),'methods':summary,
        'deployment_id':protocol['deployment_id'],'run':str(run)}
    write(run/'progress.json',result)
    if final:
        result['finished_at_utc'] = utc()
        write(run/'summary.json',result)
        write(run/'per_question.json',predictions)
        lines = ['# Full LongBench v2 baseline evaluation','',
            'All 125 Multi-Document QA IDs per method. Failures count as incorrect; no slow-case exclusions.',
            '', '| Method | IDs | Valid answers | Failures | Correct / 125 | Accuracy |',
            '|---|---:|---:|---:|---:|---:|']
        for method,x in summary.items():
            lines.append(f'| {method} | {x["terminal"]} | {x["valid_answers"]} | {x["errors"]} | {x["correct"]} | {x["accuracy_all_ids"]:.1%} |')
        lines.extend(['','Weights: identical Qwen3.5-0.8B Q4_K_M; thinking/MTP disabled. Same reader prompt and 128-token final grammar.',
            'Frozen MiniLM CPU encoder and offline construction cost are additional resources. This is an adapted implementation, not paper-default-model reproduction.',
            'Parallel replica execution changes scheduling and may change numerical sampling; historical single-server results are not a fresh matched-throughput control.',
            'Raw inputs, responses, transport attempts, source and configuration hashes remain in this run directory.',
            'See protocol.json, summary.json and per_question.json for complete identities, timings and failure details.'])
        (run/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        if protocol.get('publish_report',True):
            report = Path(protocol['repository'])/'docs/BASELINE_FULL_RESULTS.md'
            report.write_text('\n'.join(lines)+f'\n\nRun: {run}\n',encoding='utf-8')
    return result


def coordinate(protocol_path):
    protocol = read(protocol_path)
    verify_sources(protocol,protocol['snapshot'])
    if sha(protocol['dataset']) != protocol['frozen_items_sha256']:
        raise RuntimeError('Frozen dataset changed')
    preflight(protocol)
    run = Path(protocol['run'])
    lock = (run/'coordinator.lock').open('a+b')
    if lock.tell()==0:
        lock.write(b'0');lock.flush()
    lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    env = dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONHASHSEED='42',NUMBA_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',REPAR_REPOSITORY=protocol['repository'],
        REPAR_BASELINE_REGISTRY=str(ROOT/'baselines'))
    gateway_log = (run/'gateway.stdout.log').open('a',encoding='utf-8')
    port = protocol['gateway_port']
    gateway = subprocess.Popen([sys.executable,'-X','utf8','-m','repar.inference_gateway',
        '--endpoints',*protocol['endpoints'],'--slots',str(protocol['endpoint_slots']),
        '--port',str(port),'--output',str(run/'transport'),'--timeout',str(protocol['call_seconds']),
        '--attempts',str(protocol['transport_attempts'])],cwd=ROOT,env=env,
        stdout=gateway_log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    write(run/'processes.json',{'coordinator':os.getpid(),'gateway':gateway.pid,'started_at_utc':utc()})
    try:
        for _ in range(60):
            if gateway.poll() is not None:
                raise RuntimeError('Gateway exited; inspect gateway.stdout.log')
            try:
                if requests.get(f'http://127.0.0.1:{port}/health',timeout=5).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Gateway failed to become healthy')
        stop = threading.Event()
        def method_stream(method):
            for item in protocol['items']:
                if stop.is_set():
                    return
                case = run/method/item['id']
                if (case/'final.json').exists():
                    continue
                attempts = case/'attempts'
                attempts.mkdir(parents=True,exist_ok=True)
                attempt = attempts/f'{len(list(attempts.iterdir()))+1:04}'
                attempt.mkdir()
                started = time.perf_counter()
                with (attempt/'stdout.log').open('w',encoding='utf-8') as log:
                    child = subprocess.Popen([sys.executable,'-X','utf8','-m',
                        'baselines.evaluation.longbench_full','--worker',str(protocol_path),method,
                        item['id'],str(attempt)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    write(case/'active.json',{'pid':child.pid,'attempt':str(attempt),'started_at_utc':utc()})
                    while child.poll() is None:
                        if stop.wait(1):
                            child.terminate()
                            child.wait(timeout=15)
                            return
                result = read(attempt/'result.json') if (attempt/'result.json').exists() else {
                    'method':method,'id':item['id'],'length':item['length'],'status':'worker_crash',
                    'correct':False,'returncode':child.returncode}
                if result.get('status')=='running':
                    result.update(status='worker_crash',correct=False,returncode=child.returncode)
                if result.get('status')=='worker_crash':
                    stop.set()
                    write(attempt/'crash.json',result)
                    raise RuntimeError(f'Infrastructure crash for {method}/{item["id"]}; no fake benchmark zero is published')
                result.update(end_to_end_seconds=time.perf_counter()-started,attempt=str(attempt))
                write(case/'final.json',result)
                print(f'CASE {method} {item["id"]} {result["status"]} {result["end_to_end_seconds"]:.1f}s',flush=True)
        with ThreadPoolExecutor(max_workers=protocol['case_workers']) as pool:
            pending = {pool.submit(method_stream,method) for method in protocol['methods']}
            while pending:
                done,pending = wait(pending,timeout=30)
                for future in done:
                    future.result()
                status = summarize(protocol)
                try:
                    state = requests.get(f'http://127.0.0.1:{port}/stats',timeout=10).json()
                    write(run/'transport_progress.json',state)
                except requests.RequestException:
                    state = {}
                print(json.dumps({'event':'progress','terminal':status['terminal'],'expected':status['expected'],
                    'methods':{m:{k:r[k] for k in ('terminal','valid_answers','errors')} for m,r in status['methods'].items()},
                    'transport':state},ensure_ascii=False),flush=True)
        result = summarize(protocol,final=True)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        gateway.terminate()
        gateway.wait(timeout=30)
        gateway_log.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/baseline_full.json')
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--report')
    parser.add_argument('--worker',nargs=4)
    args = parser.parse_args()
    if args.worker:
        case_worker(*args.worker)
        return
    if args.report:
        print(json.dumps(summarize(read(args.report),final=True),ensure_ascii=False))
        return
    if args.resume:
        protocol_path = Path(args.resume).resolve()
        protocol = read(protocol_path)
        snapshot = Path(protocol['snapshot']).resolve()
        if ROOT.resolve() != snapshot:
            # A repository entry point must resume the frozen executable, even
            # after later maintenance edits have changed the working checkout.
            env = dict(os.environ,PYTHONPATH=str(snapshot/'src'),PYTHONHASHSEED='42')
            completed = subprocess.run([sys.executable,'-X','utf8','-m',
                'baselines.evaluation.longbench_full','--resume',str(protocol_path)],cwd=snapshot,env=env)
            raise SystemExit(completed.returncode)
        return coordinate(protocol_path)
    output = args.output or ('artifacts/baseline_full_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    protocol_path = prepare(args.config,output)
    protocol = read(protocol_path)
    env = dict(os.environ,PYTHONPATH=str(Path(protocol['snapshot'])/'src'),PYTHONHASHSEED='42')
    print('FROZEN '+str(protocol_path),flush=True)
    result = subprocess.run([sys.executable,'-X','utf8','-m','baselines.evaluation.longbench_full',
        '--resume',str(protocol_path)],cwd=protocol['snapshot'],env=env)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
