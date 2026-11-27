"""Activate v2 CLI defaults after the completed, audited experiment only."""
import hashlib,time
from pathlib import Path
from repar.io import read_json,write_json

assert Path('artifacts/v2_complete.json').exists()
assert read_json('artifacts/experiment_audit_v2.json')['all_checks_passed']
path=Path('src/repar/cli.py')
original=path.read_bytes()
frozen=Path('artifacts/versions/v2_inference/cli.py')
frozen.parent.mkdir(parents=True,exist_ok=True)
if not frozen.exists():frozen.write_bytes(original)
original=frozen.read_bytes()
updated=original.replace(b'default="configs/experiment.json"',b'default="configs/repar_v2.json"').replace(
    b'default="artifacts/experiment"',b'default="artifacts/experiment_v2"')
assert updated!=original
write_json('artifacts/v2_default_routing.json',{'time':time.strftime('%FT%T%z'),
    'frozen_cli':str(frozen),'frozen_sha256':hashlib.sha256(original).hexdigest(),
    'current_sha256':hashlib.sha256(updated).hexdigest(),'after_complete_experiments':True,
    'change':'Only CLI default config/output paths switch to v2. Recorded runs passed explicit paths; inference behavior is unchanged.'})
path.write_bytes(updated)
print('Default CLI now uses configs/repar_v2.json and artifacts/experiment_v2')
