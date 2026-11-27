"""Read-only inference audit and final report watcher. Never generates answers."""
from __future__ import annotations
import argparse
import collections
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def wilson(correct,total):
    z=1.959963984540054
    p=correct/total
    scale=1+z*z/total
    center=(p+z*z/(2*total))/scale
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/scale
    return [max(0,center-radius),min(1,center+radius)]


def paired(first,second):
    assert first.keys()==second.keys()
    a=sum(first[k] and not second[k] for k in first)
    b=sum(second[k] and not first[k] for k in first)
    n=a+b
    p=min(1.0,2*sum(math.comb(n,k) for k in range(min(a,b)+1))/2**n) if n else 1.0
    return {'first_only_correct':a,'second_only_correct':b,'exact_two_sided_p':p}


def finalize(directory,historical=None):
    run=Path(directory)
    protocol=read(run/'protocol.json')
    ids={x['id'] for x in protocol['items']}
    items={x['_id']:x for x in read(protocol['dataset'])}
    assert set(items)==ids
    assert sha(protocol['dataset'])==protocol['frozen_items_sha256']
    for relative,expected in protocol['source_hashes'].items():
        assert sha(Path(protocol['snapshot'])/relative)==expected,relative
    records={}
    scores={}
    metrics={}
    cache_files=0
    maximum_prompt=0
    request_caps=collections.Counter()
    cache_grammar=0
    for method in protocol['methods']:
        rows=[read(path) for path in (run/method).glob('*/final.json')]
        assert len(rows)==len(ids) and {x['id'] for x in rows}==ids
        for row in rows:
            assert row['status']!='running'
            expected=row['status']=='completed' and row.get('pred')==items[row['id']]['answer']
            assert bool(row.get('correct'))==bool(expected)
            assert row.get('context_sha256')==next(x['context_sha256'] for x in protocol['items'] if x['id']==row['id'])
        records[method]=rows
        scores[method]={x['id']:bool(x.get('correct')) for x in rows}
        errors=collections.Counter(x.get('error_type',x['status']) for x in rows if x['status']!='completed')
        interval=wilson(sum(scores[method].values()),len(ids))
        logical_tokens=collections.Counter()
        for row in rows:
            for key in ('index_llm','answer_llm'):
                stats=row.get(key,{})
                for field in ('calls','cache_hits','prompt_tokens','completion_tokens','length_finishes'):
                    logical_tokens[field]+=stats.get(field,0)
        metrics[method]={'ids':len(rows),'valid_answers':sum(x['status']=='completed' for x in rows),
            'correct':sum(scores[method].values()),'accuracy':sum(scores[method].values())/len(ids),
            'wilson_95':interval,'errors':dict(errors),'logical_llm':dict(logical_tokens),
            'case_wall_seconds_sum':sum(x['end_to_end_seconds'] for x in rows),
            'case_wall_seconds_median':statistics.median(x['end_to_end_seconds'] for x in rows),
            'build_seconds_completed_sum':sum(x.get('build_seconds',0) for x in rows),
            'embedding_seconds_sum':sum(x.get('embedding',{}).get('seconds',0) for x in rows),
            'hippo_ppr_cases':sum(x.get('index_stats',{}).get('retrieval_mode')=='ppr' for x in rows),
            'hippo_dense_fallback_cases':sum(x.get('index_stats',{}).get('retrieval_mode')=='upstream_dense_fallback' for x in rows)}
        for path in (run/method).glob('*/cache/*.json'):
            cached=read(path)
            body=cached['request']
            identity={'request':body,'deployment_id':protocol['deployment_id']}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            assert key==path.stem==cached['key']
            assert all(body[k]==protocol['model_config'][k] for k in ('model','temperature','seed'))
            assert body['chat_template_kwargs']['enable_thinking'] is False
            if 'grammar' in body:
                assert body['grammar']==protocol['model_config']['answer_grammar']
                assert body['max_tokens']==protocol['model_config']['answer_max_tokens']
                cache_grammar+=1
            request_caps[body['max_tokens']]+=1
            response=cached['response']
            transport=response['_repar_transport']
            request_id=transport['request_id']
            raw=read(run/'transport/calls'/request_id[:2]/f'{request_id}_{transport["attempts"]}.json')
            assert raw['request']==body and raw['endpoint']==transport['endpoint']
            assert raw['response']=={k:v for k,v in response.items() if k!='_repar_transport'}
            maximum_prompt=max(maximum_prompt,response.get('usage',{}).get('prompt_tokens',0))
            cache_files+=1
    physical=collections.Counter()
    physical_by_method=collections.defaultdict(collections.Counter)
    routes=collections.Counter()
    backend_errors=collections.Counter()
    for path in (run/'transport/calls').glob('*/*.json'):
        row=read(path)
        method=row['case_id'].split('/')[0]
        usage=row.get('response',{}).get('usage',{}) if isinstance(row.get('response'),dict) else {}
        physical['attempts']+=1
        physical['prompt_tokens']+=usage.get('prompt_tokens',0)
        physical['completion_tokens']+=usage.get('completion_tokens',0)
        physical_by_method[method]['attempts']+=1
        physical_by_method[method]['completion_tokens']+=usage.get('completion_tokens',0)
        routes[row['endpoint']]+=1
        if row['status']>=400:
            backend_errors[str(row['status'])]+=1
    comparisons={}
    if historical:
        history=[json.loads(line) for line in Path(historical).read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len(history)==len(ids) and {x['_id'] for x in history}==ids
        for row in history:
            assert row['context_sha256']==next(x['context_sha256'] for x in protocol['items'] if x['id']==row['_id'])
            assert row['answer']==items[row['_id']]['answer']
        reference={x['_id']:x.get('pred')==x['answer'] for x in history}
        for method,value in scores.items():
            comparisons[method+'_vs_historical_RePAR_v2']=paired(value,reference)
        reference_metric={'correct':sum(reference.values()),'ids':len(reference),
                          'accuracy':sum(reference.values())/len(reference),
                          'path':str(historical),'sha256':sha(historical)}
    else:
        reference_metric=None
    audit={'status':'passed','unique_ids_per_method':len(ids),
        'methods':len(records),'terminal_items':sum(len(x) for x in records.values()),
        'cache_files_verified':cache_files,'final_grammar_cache_records':cache_grammar,
        'maximum_observed_prompt_tokens':maximum_prompt,'request_token_caps':dict(request_caps),
        'physical_transport':dict(physical),'physical_by_method':{k:dict(v) for k,v in physical_by_method.items()},
        'routes':dict(routes),'backend_error_statuses':dict(backend_errors),
        'metrics':metrics,'paired_comparisons':comparisons,'historical_reference':reference_metric,
        'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    write(run/'final_audit.json',audit)
    lines=['# 三种Baseline完整评估结果','',
        '范围：LongBench v2 Multi-Document QA，三个方法各125个唯一ID。失败按错误计入125分母，未删除慢题。','',
        '| 方法 | 成功回答 | 失败 | 正确/125 | 准确率 | Wilson 95%区间 |',
        '|---|---:|---:|---:|---:|---:|']
    for method,x in metrics.items():
        lines.append(f'| {method} | {x["valid_answers"]} | {x["ids"]-x["valid_answers"]} | {x["correct"]} | {x["accuracy"]:.1%} | {x["wilson_95"][0]:.1%}–{x["wilson_95"][1]:.1%} |')
    if reference_metric:
        lines.extend(['',f'历史RePAR v2：{reference_metric["correct"]}/125，{reference_metric["accuracy"]:.1%}。这是已有结果复用，未在双服务上重新评估，不能作同硬件并发条件的耗时对照。'])
    lines.extend(['','| 方法 | 逐题墙钟之和(小时) | 逐题中位(秒) | 逻辑LLM调用 | 客户端缓存命中 | 长度终止 |',
        '|---|---:|---:|---:|---:|---:|'])
    for method,x in metrics.items():
        llm=x['logical_llm']
        lines.append(f'| {method} | {x["case_wall_seconds_sum"]/3600:.2f} | {x["case_wall_seconds_median"]:.1f} | {llm.get("calls",0)} | {llm.get("cache_hits",0)} | {llm.get("length_finishes",0)} |')
    lines.extend(['','逐题墙钟包含建库、排队、检索和作答；三个条件并行，因此不能将这些时长相加后称作完整实验墙钟。缓存命中不视作真实推理加速。',
        f'实际传输尝试{physical["attempts"]}次；完整响应中的prompt/completion tokens为{physical["prompt_tokens"]:,}/{physical["completion_tokens"]:,}，无完整响应的失败请求未估算其未知生成量。',
        '底座为相同Qwen3.5-0.8B标准Q4、关闭thinking和MTP。最终三个reader使用同一官方RAG模板、128-token上限和答案grammar。',
        '额外冻结MiniLM CPU encoder、离线摘要/抽取和不同top_k语义/证据预算均为显式资源差异；这不是论文默认强模型配置复现。',
        '双服务各8槽位、每槽65536上下文；并行执行可能改变数值采样。观察到的最大prompt tokens见审计；没有静默截断或换成其他retriever。',
        '三道先前pilot题已被观察；没有根据测试答案调prompt或挑样本。',
        '', '错误类型、真实图使用、逐题证据、配对精确检验及完整请求审计见final_audit.json与per_question.json。',
        f'',f'原始运行目录：{run}',f'冻结协议：{run/"protocol.json"}'])
    report='\n'.join(lines)+'\n'
    (run/'REPORT.md').write_text(report,encoding='utf-8')
    (Path(protocol['repository'])/'docs/BASELINE_FULL_RESULTS.md').write_text(report,encoding='utf-8')
    write(run/'report_status.json',{'status':'complete','audit':'passed','time_utc':utc_now()})
    return audit


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--historical')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    run=Path(args.run)
    if args.watch:
        while not (run/'summary.json').exists():
            write(run/'report_status.json',{'status':'waiting_for_full_coverage','checked_at_utc':utc_now()})
            time.sleep(30)
    try:
        audit=finalize(run,args.historical)
        print(json.dumps({'status':'complete','terminal_items':audit['terminal_items'],'metrics':audit['metrics']},ensure_ascii=False),flush=True)
    except Exception:
        write(run/'report_status.json',{'status':'audit_failed','error':traceback.format_exc(),'time_utc':utc_now()})
        raise


if __name__=='__main__':
    main()
