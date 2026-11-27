"""Verify an installed wheel from outside the checkout without generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repository', required=True)
parser.add_argument('--target', required=True)
parser.add_argument('--wheel', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
repo, target, wheel = (Path(x).resolve() for x in (args.repository, args.target, args.wheel))
sys.path.insert(0, str(target))
os.environ['REPAR_REPOSITORY'] = str(repo)

import repar
import baselines
assert Path(repar.__file__).is_relative_to(target)
assert Path(baselines.__file__).is_relative_to(target)
from baselines._upstream import upstream, module
from baselines.RaptorRAG import RaptorRetriever
from baselines.HippoRAG2 import HippoRetriever
from baselines.HyperGraphRAG import HyperGraphRetriever
from baselines.RaptorRAG.adapters.longbench import answer
from baselines.HippoRAG2.adapters.longbench import answer as hippo_answer
from baselines.HyperGraphRAG.adapters.longbench import answer as hyper_answer
assert answer is hippo_answer is hyper_answer
loaded = {}
for method, path, component in [('RaptorRAG','raptor','tree_structures'),
        ('HippoRAG2','src/hipporag','utils.misc_utils'), ('HyperGraphRAG','hypergraphrag','utils')]:
    namespace, spec = upstream(method, path)
    module(namespace, component)
    loaded[method] = spec['commit']
from baselines.evaluation.longbench import prompt
assert 'Mira' in prompt({'question':'Who operates Lumen?', **{'choice_'+x:'station '+x for x in 'ABCD'}}, 'Mira operates Lumen.')
from repar.model_support import SentenceEncoder
encoder = SentenceEncoder(repo / 'models/all-MiniLM-L6-v2')
vector = encoder.encode(['Mira operates the Lumen observatory. ' * 100])
assert vector.shape == (1,384) and encoder.calls[-1]['windows'] > 1
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    assert 'baselines/evaluation/0shot_rag.txt' in names
    assert not any(x.endswith('.pdf') or 'third_party/' in x or 'models/' in x for x in names)
    assert archive.read('baselines/evaluation/0shot_rag.txt') == (repo/'third_party/prompts/0shot_rag.txt').read_bytes()
result = {'status':'passed', 'cwd':str(Path.cwd()), 'isolated_python':bool(sys.flags.isolated),
    'installed_baselines':str(Path(baselines.__file__)), 'installed_repar':str(Path(repar.__file__)),
    'source_commits':loaded, 'encoder_shape':list(vector.shape), 'encoder_windows':encoder.calls[-1]['windows'],
    'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(), 'wheel_bytes':wheel.stat().st_size}
Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
