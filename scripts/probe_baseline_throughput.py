"""Measure real output throughput on independent requests, without benchmark data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoints',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--parallel',type=int,default=16)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    phases = []
    for phase,workers in [('serial_1',1),('parallel_1',args.parallel),
                          ('parallel_2',args.parallel),('serial_2',1)]:
        started = time.perf_counter()
        def run(index):
            endpoint = args.endpoints[index%len(args.endpoints)] if workers>1 else args.endpoints[0]
            body = {'model':'Qwen3.5-0.8B-Q4_K_M', 'messages':[{'role':'user','content':
                f'Prepare detailed notes for workshop station {index}. Explain how to preserve provenance, '
                'version records, and reproducible measurements in a general document archive. '
                'Continue with concrete technical examples and limitations.'}],
                'temperature':0.1,'seed':42,'max_tokens':512,'stream':False,
                'chat_template_kwargs':{'enable_thinking':False}}
            begin = time.perf_counter()
            response = requests.post(endpoint+'/v1/chat/completions',json=body,timeout=(10,300))
            response.raise_for_status()
            value = response.json()
            record = {'index':index,'endpoint':endpoint,'request':body,'response':value,
                      'seconds':time.perf_counter()-begin}
            (out/f'{phase}_{index:02}.json').write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8')
            return record
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(run,range(16)))
        elapsed = time.perf_counter()-started
        tokens = sum(x['response']['usage']['completion_tokens'] for x in records)
        result = {'phase':phase,'concurrency':workers,'seconds':elapsed,
                  'completion_tokens':tokens,'output_tokens_per_second':tokens/elapsed}
        phases.append(result)
        print(json.dumps(result),flush=True)
    serial = statistics.mean(x['output_tokens_per_second'] for x in phases if x['concurrency']==1)
    parallel = statistics.mean(x['output_tokens_per_second'] for x in phases if x['concurrency']>1)
    result = {'phases':phases,'measured_throughput_ratio':parallel/serial,
        'limitations':'Independent synthetic requests, no client cache; existing math task shares the GPU. '
                       'S-P-P-S order reduces warmup bias; not a benchmark accuracy experiment.'}
    try:
        result['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader'],text=True).strip()
    except Exception:
        pass
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
