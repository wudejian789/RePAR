"""Read real full-run progress, including active long-running construction."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil

import requests


def tail_text(path,limit=32768):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-limit))
        return stream.read().decode('utf-8',errors='replace')


def progress(directory):
    run=Path(directory)
    protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    state=json.loads((run/'progress.json').read_text(encoding='utf-8'))
    active=[]
    for method in protocol['methods']:
        for marker in (run/method).glob('*/active.json'):
            if (marker.parent/'final.json').exists():
                continue
            item=json.loads(marker.read_text(encoding='utf-8'))
            attempt=Path(item['attempt'])
            result_path=attempt/'result.json'
            result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
            journal=attempt/'calls.jsonl'
            if journal.exists():
                with journal.open(encoding='utf-8') as stream:
                    calls=sum(1 for _ in stream)
            else:
                calls=0
            log=tail_text(attempt/'stdout.log')
            bars=re.findall(r'([^\r\n:]{1,65}):\s*(\d+)%\|[^\r\n]*?\|\s*(\d+)/(\d+)',log)
            stage=None
            if bars:
                name,percent,done,total=bars[-1]
                stage={'name':name.strip(),'percent':int(percent),'done':int(done),'total':int(total)}
            active.append({'method':method,'id':marker.parent.name,'phase':result.get('phase','starting'),
                'words':result.get('words'),'pid':item['pid'],
                'elapsed_seconds':(datetime.now(timezone.utc)-datetime.fromisoformat(item['started_at_utc'])).total_seconds(),
                'journal_events':calls,'stage':stage,
                'last_log_age_seconds':max(0,datetime.now(timezone.utc).timestamp()-(attempt/'stdout.log').stat().st_mtime)
                    if (attempt/'stdout.log').exists() else None,
                'last_journal_age_seconds':max(0,datetime.now(timezone.utc).timestamp()-journal.stat().st_mtime)
                    if journal.exists() else None})
    state['active_cases']=active
    try:
        state['gateway']=requests.get(f'http://127.0.0.1:{protocol["gateway_port"]}/stats',timeout=5).json()
    except requests.RequestException:
        state['gateway']='not_listening'
    state['disk_free_gb']=round(shutil.disk_usage(run).free/1024**3,1)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='artifacts/baseline_full_qwen08_dual_v1')
    args=parser.parse_args()
    print(json.dumps(progress(args.run),ensure_ascii=False,indent=2))
