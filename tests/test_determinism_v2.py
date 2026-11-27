"""Graph observations must be byte-equivalent across Python hash seeds."""
import json
import os
import pickle
import subprocess
import sys
import pytest

pytest.importorskip("en_core_web_sm",reason="Install the pinned English extraction model to test v2")

from repar.index_v2 import KnowledgeIndex


def test_graph_observations_are_independent_of_process_hash_seed(tmp_path):
    config={"chunk_words":40,"tree_fanout":4,"graph_neighbors":3,"index_version":"entity-graph-v2","graph_model":"en_core_web_sm"}
    text=("The Delta Commission developed the Silver Protocol to monitor river pollution. "
          "Harbor Council adopted the Silver Protocol. Harbor Council funds mobile clinics. "
          "Mobile clinics reduce travel costs. North Agency did not adopt the Silver Protocol.")
    index=KnowledgeIndex(text,config)
    path=tmp_path/'index.pkl'
    with path.open('wb') as f:pickle.dump(index,f)
    code="""import pickle,json,sys
with open(sys.argv[1],'rb') as f: kb=pickle.load(f)
entities=kb.graph_entities('Silver Protocol Harbor Council',8)
eid=next(e['id'] for e in kb.entity_graph.entities if e['canonical']=='silver protocol')
neighbors=kb.graph_neighbors([eid],query='who adopted and developed Silver Protocol',hops=2)
print(json.dumps([entities,neighbors],sort_keys=True))
"""
    outputs=[]
    for seed in ('7','41'):
        outputs.append(subprocess.check_output([sys.executable,'-c',code,str(path)],
                       env={**os.environ,'PYTHONHASHSEED':seed},text=True))
    assert outputs[0]==outputs[1]
