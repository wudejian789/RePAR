import json
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import pytest

from repar.concurrent_client import ConcurrentLLMClient
from repar.inference_gateway import Gateway, serve
from baselines.evaluation.longbench_full import summarize, write


def test_parallel_client_keeps_thread_local_response_and_real_cache(tmp_path):
    lock = threading.Lock()
    count = [0]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            value = req['messages'][-1]['content']
            time.sleep(.02)
            with lock:
                count[0] += 1
            body = json.dumps({'choices':[{'message':{'content':value},'finish_reason':value}],
                               'usage':{'prompt_tokens':1,'completion_tokens':1}}).encode()
            self.send_response(200)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Repar-Endpoint','fixture-backend')
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    client = ConcurrentLLMClient({'base_url':f'http://127.0.0.1:{server.server_port}',
        'model':'fixture','temperature':.1,'seed':42,'enable_thinking':False,
        'deployment_id':'fixture-v1','answer_grammar':'not-for-indexing'},tmp_path/'cache',journal=tmp_path/'calls.jsonl')
    try:
        def call(i):
            value = str(i)
            response = client.complete([{'role':'user','content':value}],tag=value)
            assert response==value
            assert client.last_call['response']['choices'][0]['finish_reason']==value
            return response
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(call,range(12)))==list(map(str,range(12)))
        assert count[0]==12
        call(0)
        assert count[0]==12 and client.calls[-1]['cached'] is True
        events = [json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
        assert len(events)==13
        assert all('grammar' not in json.loads(p.read_text())['request'] for p in (tmp_path/'cache').glob('*.json'))
    finally:
        server.shutdown();server.server_close()


def test_gateway_global_slot_limit_and_attempt_trace(tmp_path, monkeypatch):
    import repar.inference_gateway as module
    active = {}; peak = {}; guard = threading.Lock()
    class Response:
        status_code = 200
        def json(self):
            return {'usage':{'completion_tokens':1}}
    def post(url,**kwargs):
        with guard:
            active[url]=active.get(url,0)+1
            peak[url]=max(peak.get(url,0),active[url])
        time.sleep(.02)
        with guard:
            active[url]-=1
        return Response()
    monkeypatch.setattr(module.requests,'post',post)
    gateway = Gateway(['http://first','http://second'],2,tmp_path,timeout=3)
    with ThreadPoolExecutor(max_workers=10) as pool:
        results=list(pool.map(lambda i:gateway.handle('/v1/chat/completions',{'id':i},'fixture'),range(20)))
    assert all(x[0]==200 for x in results)
    assert max(peak.values())<=2
    assert gateway.completed==20 and gateway.leases.qsize()==4
    assert len(list((tmp_path/'calls').rglob('*.json')))==20


def test_full_report_requires_every_id_and_counts_failures(tmp_path):
    protocol={'run':str(tmp_path),'repository':str(tmp_path),'methods':['A'],
              'items':[{'id':'one'},{'id':'two'}],'deployment_id':'fixture','publish_report':False}
    write(tmp_path/'A/one/final.json',{'id':'one','method':'A','status':'completed','correct':True})
    with pytest.raises(RuntimeError,match='Incomplete'):
        summarize(protocol,final=True)
    write(tmp_path/'A/two/final.json',{'id':'two','method':'A','status':'error','correct':False})
    result=summarize(protocol,final=True)
    assert result['status']=='complete'
    assert result['methods']['A']['accuracy_all_ids']==.5
    assert result['methods']['A']['errors']==1


def test_final_audit_publishes_only_full_coverage(tmp_path):
    import importlib.util
    from pathlib import Path
    import hashlib
    script=Path(__file__).resolve().parents[1]/'scripts/finalize_baseline_full.py'
    spec=importlib.util.spec_from_file_location('fixture_final_audit',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path/'docs').mkdir()
    items=[{'_id':'one','answer':'A'},{'_id':'two','answer':'B'}]
    write(tmp_path/'input/items.json',items)
    digest=hashlib.sha256((tmp_path/'input/items.json').read_bytes()).hexdigest()
    protocol={'run':str(tmp_path),'repository':str(tmp_path),'snapshot':str(tmp_path/'source'),
        'dataset':str(tmp_path/'input/items.json'),'frozen_items_sha256':digest,'source_hashes':{},
        'items':[{'id':'one','context_sha256':'abc'},{'id':'two','context_sha256':'def'}],
        'methods':['A'],'deployment_id':'fixture'}
    write(tmp_path/'protocol.json',protocol)
    write(tmp_path/'A/one/final.json',{'method':'A','id':'one','status':'completed','pred':'A',
        'correct':True,'context_sha256':'abc','end_to_end_seconds':1})
    with pytest.raises(AssertionError):
        module.finalize(tmp_path)
    write(tmp_path/'A/two/final.json',{'method':'A','id':'two','status':'error',
        'correct':False,'context_sha256':'def','end_to_end_seconds':2})
    result=module.finalize(tmp_path)
    assert result['status']=='passed'
    assert result['metrics']['A']['accuracy']==.5
    assert (tmp_path/'docs/BASELINE_FULL_RESULTS.md').is_file()
