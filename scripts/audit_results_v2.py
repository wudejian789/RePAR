"""Verify v2 phase order, parameter binding, raw API identity and source evidence."""
import hashlib,json
from pathlib import Path
from collections import Counter

from repar.evaluation import extract_answer
from repar.io import digest,load_public_samples,read_json,read_jsonl,write_json
from repar.workflow import signature


def main():
    frozen=read_json('artifacts/implementation_manifest_v2.json')
    for filename,sha in frozen['files'].items():
        if Path(filename).parts[0]=='tests':
            continue  # extra validation tests can be added without altering inference
        actual=hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if actual!=sha and Path(filename).as_posix()=='src/repar/cli.py':
            routing=read_json('artifacts/v2_default_routing.json')
            original=Path(routing['frozen_cli']).read_bytes()
            assert hashlib.sha256(original).hexdigest()==sha
            expected=original.replace(b'default="configs/experiment.json"',b'default="configs/repar_v2.json"').replace(
                b'default="artifacts/experiment"',b'default="artifacts/experiment_v2"')
            assert Path(filename).read_bytes()==expected
            assert routing['after_complete_experiments'] and actual==routing['current_sha256']
        else:
            assert actual==sha,filename
    metadata=read_json('artifacts/experiment_v2/run_metadata.json')
    config=metadata['config']
    assert config['model']=='Qwen3.5-0.8B-Q4_K_M' and config['agent_version']=='react-v2'
    samples={x['_id']:x for x in load_public_samples('data/longbench_v2.json')}
    counts=Counter()
    for method in ('repar','repar_tree'):
        rows=read_jsonl(f'artifacts/experiment_v2/{method}.jsonl')
        assert len(rows)==len(samples)==len({r['_id'] for r in rows})
        assert {r['_id'] for r in rows}==set(samples)
        for row in rows:
            item=samples[row['_id']]
            assert row['context_sha256']==digest(item['context'])
            assert row['config_sha256']==metadata['config_sha256']
            assert row['answer']==item['answer']
            assert row['pred']==extract_answer(row['response'])
            assert row['judge']==(row['pred']==item['answer'])
            trace=read_json(row['trace_path'])
            assert trace['response']==row['response']
            assert 'answer' not in trace
            completed=set()
            cursor=0
            for step in trace['trace']:
                assert step['protocol']=='plan_action_observe_reflect_v2'
                tags=[]
                for phase,raw_name in (('plan','plan_raw'),('action','action_raw'),('reflection','reflection_raw')):
                    if raw_name not in step:continue
                    raw=read_json('artifacts/cache/llm/'+row['calls'][cursor]['key']+'.json')
                    tag=row['calls'][cursor]['tag']
                    assert tag.startswith('v2_'+phase+'_')
                    # Output repair is permitted but must precede the action.
                    response=raw['response']['choices'][0]['message']['content']
                    cursor+=1
                    repairs=0
                    while response!=step[raw_name]:
                        repairs+=1
                        assert repairs<=2
                        raw=read_json('artifacts/cache/llm/'+row['calls'][cursor]['key']+'.json')
                        assert row['calls'][cursor]['tag'].startswith('v2_'+phase+'_')
                        assert '_repair' in row['calls'][cursor]['tag']
                        response=raw['response']['choices'][0]['message']['content']
                        cursor+=1
                    if phase=='reflection':
                        payload=json.loads(raw['request']['messages'][1]['content'])
                        assert payload['action']['name']==step['executed'][0]['name']
                        assert payload['observation']['operator']==step['executed'][0]['observation']['operator']
                for call in step.get('executed',[]):
                    sig=signature(call)
                    assert sig not in completed
                    completed.add(sig)
                    selected=[a for a in step['available_actions'] if a['name']==call['name'] and a['parameters']==call['parameters']]
                    assert selected
                    if method=='repar_tree':assert not call['name'].startswith('graph_')
                    for record in call['observation'].get('results',[]):
                        assert record['text']==item['context'][record['start']:record['end']]
                        counts['source_records']+=1
                    counts['tool_calls']+=1
            assert cursor==len(row['calls'])-1
            for call in row['calls']:
                raw=read_json('artifacts/cache/llm/'+call['key']+'.json')
                identity={'request':raw['request'],'deployment_id':config['deployment_id']}
                assert digest(json.dumps(identity,sort_keys=True,ensure_ascii=False))==call['key']
                assert raw['response']['usage']==call['usage']
                assert raw['request']['model']==config['model']
                counts['raw_api_records']+=1
            assert raw['response']['choices'][0]['message']['content']==row['response']
            counts['predictions']+=1
    write_json('artifacts/experiment_audit_v2.json',{'all_checks_passed':True,'counts':dict(counts),
                                                    'checks':['full125 coverage','frozen inference','phase order and actual observation',
                                                              'bound parameters','no duplicate execution','raw cache identity','exact source spans']})
    print('V2 RESULT AUDIT PASSED',dict(counts),flush=True)


if __name__=='__main__':main()
