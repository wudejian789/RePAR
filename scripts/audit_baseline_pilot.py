"""Audit frozen source, unique coverage and real request identities; no generation."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from repar.io import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='artifacts/baseline_pilot_20261118_v1')
    parser.add_argument('--output', default='artifacts/maintenance/baseline_pilot_audit.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = Path(args.run)
    protocol = json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    rows = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    expected = {(method, sample['id']) for method in protocol['methods'] for sample in protocol['samples']}
    assert len(rows) == len(expected)
    assert {(r['method'],r['id']) for r in rows} == expected
    changed = [rel for rel,sha in protocol['source_hashes'].items()
               if hashlib.sha256((root/rel).read_bytes()).hexdigest() != sha]
    assert not changed, changed
    calls = []
    for journal in run.glob('*/*/calls.jsonl'):
        for line in journal.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            if event.get('event') != 'completion':
                continue
            assert not event['cached']
            cached = json.loads((journal.parent/'cache'/(event['key']+'.json')).read_text(encoding='utf-8'))
            assert cached['response'] == event['response']
            request = cached['request']
            cfg = protocol['model_config']
            assert all(request[k] == cfg[k] for k in ('model','temperature','seed'))
            assert request['chat_template_kwargs']['enable_thinking'] is False
            assert digest(json.dumps({'request':request,'deployment_id':cfg['deployment_id']},
                                     sort_keys=True,ensure_ascii=False)) == event['key']
            if event['tag'] == 'baseline_longbench_answer':
                assert request['grammar'] == cfg['answer_grammar']
                assert request['max_tokens'] == cfg['answer_max_tokens']
            else:
                assert 'grammar' not in request
            calls.append(event)
    result = {'status':'passed','rows':len(rows),'unique_method_ids':len(expected),
        'statuses':dict(collections.Counter(r['status'] for r in rows)),
        'journal_completions':len(calls),'cache_hits':0,
        'frozen_source_files_verified':len(protocol['source_hashes']),
        'changed_frozen_sources':changed,
        'answer_calls':sum(x['tag']=='baseline_longbench_answer' for x in calls),
        'intermediate_calls':sum(x['tag']!='baseline_longbench_answer' for x in calls),
        'deployment':protocol['model_config']['deployment_id'],
        'all_completed_request_identities_verified':True}
    target = Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
