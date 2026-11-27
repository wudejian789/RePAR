"""Foreground observation only: report fresh progress and infrastructure stalls."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from baseline_full_progress import progress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='artifacts/baseline_full_qwen08_dual_v1')
    parser.add_argument('--interval',type=float,default=45)
    args=parser.parse_args()
    if not 1<=args.interval<=60:
        parser.error('Monitoring interval must be 1..60 seconds')
    run=Path(args.run)
    previous_calls=None
    while True:
        try:
            state=progress(run)
            gateway=state.get('gateway',{})
            completed=gateway.get('completed',0) if isinstance(gateway,dict) else None
            delta=None if completed is None or previous_calls is None else completed-previous_calls
            previous_calls=completed
            result={'time':datetime.now().astimezone().isoformat(timespec='seconds'),
                'terminal':state['terminal'],'expected':state['expected'],
                'methods':{m:{k:r[k] for k in ('terminal','valid_answers','errors')}
                           for m,r in state['methods'].items()},
                'active':state['active_cases'],'gateway':gateway,'call_delta':delta,
                'disk_free_gb':state['disk_free_gb']}
            if state['status']=='complete':
                report=run/'report_status.json'
                result['report']=json.loads(report.read_text(encoding='utf-8')) if report.exists() else None
                print(json.dumps(result,ensure_ascii=False),flush=True)
                if result['report'] and result['report'].get('status') in ('complete','audit_failed'):
                    return
            else:
                print(json.dumps(result,ensure_ascii=False),flush=True)
        except Exception as exc:
            print(json.dumps({'time':datetime.now().isoformat(),'monitor_error':type(exc).__name__,
                              'error':str(exc)},ensure_ascii=False),flush=True)
        time.sleep(args.interval)


if __name__=='__main__':
    main()
