"""Adopt unchanged frozen workers and use spare case slots for unfinished methods.

This scheduler never edits protocol/source, retries a terminal case, or constructs
a model request. It only launches the original frozen --worker entry point.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def next_job(methods,items,finished,active):
    counts={m:sum(k[0]==m for k in active) for m in methods}
    remaining={m:sum((m,x['id']) not in finished and (m,x['id']) not in active for x in items) for m in methods}
    for method in sorted(methods,key=lambda m:(counts[m],-remaining[m],methods.index(m))):
        for item in items:
            key=(method,item['id'])
            if key not in finished and key not in active:
                return key
    return None


class AdoptedProcess:
    def __init__(self,pid):
        self.pid=pid
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        self.kernel.OpenProcess.restype=wintypes.HANDLE
        self.kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        self.kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        self.handle=self.kernel.OpenProcess(0x1000|0x100000,False,pid)
        if not self.handle:
            self.handle=None
    def poll(self):
        if not self.handle:
            return 0
        code=wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle,ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return None if code.value==259 else code.value
    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    args=parser.parse_args()
    protocol_path=Path(args.protocol).resolve()
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    snapshot=Path(protocol['snapshot'])
    sys.path[:0]=[str(snapshot),str(snapshot/'src')]
    from baselines.evaluation.longbench_full import read,write,verify_sources,summarize,preflight,utc
    import requests
    run=Path(protocol['run'])
    verify_sources(protocol,snapshot)
    assert hashlib.sha256(Path(protocol['dataset']).read_bytes()).hexdigest()==protocol['frozen_items_sha256']
    preflight(protocol)
    lock=(run/'coordinator.lock').open('r+b')
    lock.seek(0)
    import msvcrt
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    requests.get(f'http://127.0.0.1:{protocol["gateway_port"]}/health',timeout=10).raise_for_status()
    active={}
    finished=set()
    for method in protocol['methods']:
        for item in protocol['items']:
            key=(method,item['id']);case=run/method/item['id']
            if (case/'final.json').exists():
                finished.add(key);continue
            if (case/'active.json').exists():
                marker=read(case/'active.json')
                active[key]={'process':AdoptedProcess(marker['pid']),'marker':marker,'case':case,'log':None}
    script=Path(__file__).resolve()
    events=run/'scheduling'
    events.mkdir(exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    source=events/(stamp+'_'+script.name)
    source.write_bytes(script.read_bytes())
    write(events/(stamp+'.json'),{'type':'execution_scheduler_takeover','scheduler_pid':os.getpid(),
        'time_utc':utc(),'frozen_worker_unchanged':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'adopted':[{ 'method':m,'id':i,'pid':x['process'].pid} for (m,i),x in active.items()],
        'completed_cases_preserved':len(finished),'case_slots':protocol['case_workers'],
        'rule':'Equalize active case counts among unfinished methods; preserve declared item order within each method. Existing terminal results are never rerun.'})
    env=dict(os.environ,PYTHONPATH=str(snapshot/'src'),PYTHONHASHSEED='42',NUMBA_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',REPAR_REPOSITORY=protocol['repository'],
        REPAR_BASELINE_REGISTRY=str(snapshot/'baselines'))
    print(f'ADOPTED {len(active)} live cases; preserved {len(finished)} terminal cases',flush=True)
    last_summary=0
    try:
        while len(finished)<len(protocol['methods'])*len(protocol['items']):
            for key,task in list(active.items()):
                process=task['process'];code=process.poll()
                if code is None:
                    continue
                if task['log']:
                    task['log'].close()
                if isinstance(process,AdoptedProcess):
                    process.close()
                attempt=Path(task['marker']['attempt']);path=attempt/'result.json'
                if not path.exists():
                    raise RuntimeError(f'Infrastructure crash without result: {key}; not publishing a fake zero')
                result=read(path)
                if result['status'] not in ('completed','error','answer_parse_error'):
                    raise RuntimeError(f'Incomplete worker state for {key}: {result["status"]}')
                elapsed=(datetime.fromisoformat(result['finished_at_utc'])-datetime.fromisoformat(task['marker']['started_at_utc'])).total_seconds()
                result.update(end_to_end_seconds=elapsed,attempt=str(attempt))
                write(task['case']/'final.json',result)
                finished.add(key);del active[key]
                print(f'CASE {key[0]} {key[1]} {result["status"]} {elapsed:.1f}s',flush=True)
            while len(active)<protocol['case_workers']:
                key=next_job(protocol['methods'],protocol['items'],finished,active)
                if key is None:
                    break
                method,item_id=key;case=run/method/item_id;attempts=case/'attempts'
                attempts.mkdir(parents=True,exist_ok=True)
                attempt=attempts/f'{len(list(attempts.iterdir()))+1:04}';attempt.mkdir()
                log=(attempt/'stdout.log').open('w',encoding='utf-8')
                process=subprocess.Popen([sys.executable,'-X','utf8','-m','baselines.evaluation.longbench_full',
                    '--worker',str(protocol_path),method,item_id,str(attempt)],cwd=snapshot,env=env,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                marker={'pid':process.pid,'attempt':str(attempt),'started_at_utc':utc()}
                write(case/'active.json',marker)
                active[key]={'process':process,'marker':marker,'case':case,'log':log}
            if time.monotonic()-last_summary>=30:
                state=summarize(protocol)
                print(json.dumps({'terminal':state['terminal'],'active':[list(k) for k in active],'time_utc':utc()}),flush=True)
                last_summary=time.monotonic()
            time.sleep(2)
        summarize(protocol,final=True)
        print('FULL COVERAGE 375/375; finalizer may now audit',flush=True)
    finally:
        lock.close()


if __name__=='__main__':
    main()
