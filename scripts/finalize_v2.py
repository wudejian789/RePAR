"""Wait for the complete v2 suite, verify it, then capture a reviewable delivery."""
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import requests

from repar.io import read_json,write_json

while not Path('artifacts/v2_complete.json').exists():
    time.sleep(5)
subprocess.run([sys.executable,'-X','utf8','scripts/audit_results_v2.py'],check=True)
subprocess.run([sys.executable,'-X','utf8','-m','pytest','tests','webserver/tests','-q'],check=True)
props=requests.get('http://127.0.0.1:9091/props',timeout=10).json()
assert props['model_alias']=='Qwen3.5-0.8B-Q4_K_M'
overview=requests.get('http://127.0.0.1:8080/data/overview.json',timeout=20).json()
assert overview['case_count']==125 and 'repar_v2' in overview['methods']
files=[*Path('src').rglob('*.py'),*Path('scripts').glob('*.py'),Path('configs/repar_v2.json'),
       *Path('webserver/static').glob('*'),*Path('webserver').glob('*.py'),Path('webserver/index.html'),
       *Path('docs').glob('*.md')]
write_json('artifacts/delivery_manifest_v2.json',{'time':time.strftime('%FT%T%z'),
    'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    'experiments':'artifacts/experiment_v2/summary.json','audit':'artifacts/experiment_audit_v2.json',
    'base_model':props['model_alias'],'web_conditions':overview['methods']})
write_json('artifacts/delivery_v2_complete.json',{'time':time.strftime('%FT%T%z'),'v2_predictions':250,
    'case_count':125,'web_predictions':125*len(overview['methods']),'audit_passed':True,'model':props['model_alias']})
print('V2 DELIVERY AUDITED AND COMPLETE',flush=True)
