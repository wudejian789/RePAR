"""Full-coverage v2 results and operational comparison with preserved v1 records."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from repar.io import load_public_samples,read_json,read_jsonl,write_json
from repar.workflow import signature
from report import stats,paired


def main():
    samples=load_public_samples("data/longbench_v2.json")
    ids=[s["_id"] for s in samples]
    conditions={"direct":("artifacts/experiment","direct"),"bm25":("artifacts/experiment","bm25"),
                "repar_v1":("artifacts/experiment","repar"),"repar_tree_v1":("artifacts/experiment","repar_tree"),
                "repar_v2":("artifacts/experiment_v2","repar"),"repar_tree_v2":("artifacts/experiment_v2","repar_tree")}
    mapped={};summaries={}
    for label,(root,method) in conditions.items():
        rows=read_jsonl(Path(root)/(method+".jsonl"))
        assert len(rows)==len(ids)==len({r["_id"] for r in rows})
        assert {r["_id"] for r in rows}==set(ids)
        mapped[label]={r["_id"]:r for r in rows}
        for sample in samples:
            r=mapped[label][sample["_id"]]
            assert r["answer"]==sample["answer"] and r["judge"]==(r["pred"]==sample["answer"])
        summaries[label]={"overall":stats(rows)}
        for dimension in ("length","difficulty","sub_domain"):
            summaries[label][dimension]={value:stats([r for r in rows if r[dimension]==value])
                                         for value in sorted({r[dimension] for r in rows})}
    operations={}
    for label in ("repar_v1","repar_tree_v1","repar_v2","repar_tree_v2"):
        traces=[read_json(mapped[label][i]["trace_path"]) for i in ids]
        executions=[c for t in traces for s in t["trace"] for c in s.get("executed",[])]
        tools=Counter(c["name"] for c in executions if "name" in c and not c.get("error"))
        duplicate_executions=0
        for t in traces:
            seen=set()
            for s in t["trace"]:
                for c in s.get("executed",[]):
                    if c.get("name"):
                        sig=signature(c)
                        duplicate_executions+=sig in seen
                        seen.add(sig)
        operations[label]={"tool_counts":dict(tools),"tool_errors":sum(bool(c.get("error")) for c in executions),
                           "duplicate_call_errors":sum("Repeated successful call" in c.get("error","") for c in executions),
                           "duplicate_executions":duplicate_executions,
                           "graph_used_samples":sum(any(c.get("name","").startswith("graph_") and not c.get("error")
                                                        for s in t["trace"] for c in s.get("executed",[])) for t in traces),
                           "neighbor_used_samples":sum(any(c.get("name")=="graph_neighbors" and not c.get("error")
                                                        for s in t["trace"] for c in s.get("executed",[])) for t in traces),
                           "termination":dict(Counter(t["termination"] for t in traces)),
                           "mean_rounds":float(np.mean([mapped[label][i]["rounds"] for i in ids])),
                           "tool_seconds":sum(c.get("tool_seconds",0) for c in executions),
                           "structured_output_repair_calls":sum('_repair' in c.get('tag','') for i in ids for c in mapped[label][i]['calls']),
                           "action_selection_ids":dict(Counter(json.loads(s["action_raw"])["action_id"]
                                                      for t in traces for s in t["trace"] if s.get("action_raw"))),
                           "first_action_candidate_selections":sum(json.loads(s["action_raw"])["action_id"]==s["available_actions"][0]["action_id"]
                                                      for t in traces for s in t["trace"] if s.get("action_raw")),
                           "phase_sequence_verified":all(s.get("protocol")=="plan_action_observe_reflect_v2" and
                                                            s.get("plan_raw") and s.get("action_raw") and
                                                            (not s.get("executed") or s.get("reflection_raw"))
                                                            for t in traces for s in t["trace"]) if label.endswith("v2") else None}
    comparisons={f"repar_v2_vs_{other}":paired(mapped["repar_v2"],mapped[other],ids)
                 for other in ("direct","bm25","repar_v1","repar_tree_v2")}
    audit=read_json("artifacts/structure_audit_v2.json")
    root=Path("artifacts/experiment_v2")
    summary={"coverage":{label:len(ids) for label in conditions},"methods":summaries,"comparisons":comparisons,
             "operations":operations,"graph_audit":audit["summary"],"conditions":conditions}
    write_json(root/"summary.json",summary)
    with (root/"per_question.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["_id","answer",*[label+suffix for label in conditions for suffix in ("_pred","_correct","_seconds")]]
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for i in ids:
            row={"_id":i,"answer":mapped["direct"][i]["answer"]}
            for label in conditions:
                value=mapped[label][i]
                row.update({label+"_pred":value["pred"],label+"_correct":value["judge"],label+"_seconds":value["seconds"]})
            writer.writerow(row)
    graph=audit["summary"]
    report=["# RePAR v2 完整实验与执行审计","",
            "全部 125 道 Multi-Document QA 的 v2 RePAR 与无图对照已完成。底座为标准 Q4、关闭 MTP；保留 v1 实测基线。",
            "本轮实现真正的 plan → action → 外部 observation → reflection，树索引原文、图索引多对多实体与有出处的关系。设计与文献依据见 [REPAR_V2_DESIGN.md](REPAR_V2_DESIGN.md)。",
            f"v2 准确率 **{summaries['repar_v2']['overall']['accuracy']:.1f}%**，比 v1 高 {summaries['repar_v2']['overall']['accuracy']-summaries['repar_v1']['overall']['accuracy']:.1f} 个百分点；{'仍未达到' if summaries['repar_v2']['overall']['accuracy']<=44 else '数值上超过'}历史 GPT-4o 44.0% 参考。v2 相对无图仅差 {summaries['repar_v2']['overall']['correct']-summaries['repar_tree_v2']['overall']['correct']} 题，不能据此证明稳定的图收益。","",
            "## 准确率与成本","",
            "| 条件 | 正确/125 | 准确率 | Wilson 95% CI | 平均模型计算秒 | 平均在线实测秒 | 平均 LLM 调用 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    names={"direct":"标准 Q4 直接回答","bm25":"BM25","repar_v1":"RePAR v1","repar_tree_v1":"v1 禁用图",
           "repar_v2":"RePAR v2","repar_tree_v2":"v2 无图"}
    for label in conditions:
        s=summaries[label]["overall"]
        report.append(f"| {names[label]} | {s['correct']}/125 | {s['accuracy']:.1f}% | [{s['wilson_95'][0]:.1f}, {s['wilson_95'][1]:.1f}] | {s['mean_service_seconds']:.3f} | {s['mean_wall_seconds']:.3f} | {s['total_llm_calls']/125:.2f} |")
    report += ["","在线实测包含索引加载、查询、LLM 与本地处理；离线建图和等待建图的时间单列，不能当作零成本。稳定版重跑复用了此前相同真实请求的缓存，缓存请求的模型计算仍计入首次真实成本；在线墙钟会因此偏低，不能直接据此声称 v2 更快。直接回答/BM25 使用原标准 Q4 的125题实测结果；片段划分相同，无需重复调用。v2 的三阶段调用增加了模型生成成本，不能只比较调用次数或毫秒级图检索。","",
               "## v2 配对比较","","| 对照 | v2 差值 pp | Bootstrap 95% CI | 胜/负 | McNemar p |","|---|---:|---:|---:|---:|"]
    for other in ("direct","bm25","repar_v1","repar_tree_v2"):
        s=comparisons["repar_v2_vs_"+other]
        report.append(f"| {names[other]} | {s['difference_pp']:+.1f} | [{s['bootstrap_95'][0]:.1f}, {s['bootstrap_95'][1]:.1f}] | {s['wins']}/{s['losses']} | {s['mcnemar_exact_p']:.4f} |")
    report += ["","同题配对 bootstrap 20,000 次，seed42；未作多重比较校正。v1 结果已知，不能将这轮迭代当作完全未见测试数据的确认性实验。没有将标准答案输入规划器或索引，也没有逐题规则。","",
               "## 工具与控制循环","",
               "| 条件 | 工具成功次数 | 工具错误 | 重复执行 | 使用图的题 | 实体邻居查询的题 | 结束原因 |",
               "|---|---|---:|---:|---:|---:|---|"]
    for label,o in operations.items():
        report.append(f"| {names[label]} | {json.dumps(o['tool_counts'],ensure_ascii=False)} | {o['tool_errors']} | {o['duplicate_executions']} | {o['graph_used_samples']} | {o['neighbor_used_samples']} | {json.dumps(o['termination'],ensure_ascii=False)} |")
    report += ["","v2 的无图对照从可用工具中移除实体/邻居查询，不再以错误阻断图调用。没有最少轮数或必须使用图的规定；预算和无进展停止均为通用资源控制。各阶段原始输入/输出与实际 observation 可在新 Trace 中检查。","",
               f"工具错误为0不代表所有模型输出都一次成功：v2 使用 {operations['repar_v2']['structured_output_repair_calls']} 次结构化输出修复，无图组使用 {operations['repar_tree_v2']['structured_output_repair_calls']} 次；这些真实调用、token 和计算成本均已计入，页面也保留各阶段的尝试与截断。v2 仍有 {operations['repar_v2']['termination'].get('budget',0)} 题耗尽预算，不能把执行合法性当作证据完整或最佳规划。","",
               "动作由模型从动态绑定的参数化候选中选择，候选展示顺序按轮次轮换，以减少固定首项偏好；这不等于已经证明模型具有最优规划能力。各选择位置的计数保存在 summary.json，结合准确率和来源质量判断实际有效性。","",
               "## 实体图的结构与出处","",
               f"全部 {graph['samples']} 个知识库通过审计，共 {graph['entities']:,} 个实体、{graph['propositions']:,} 个原文陈述、{graph['relations']:,} 条抽取关系，核对 {graph['mentions_audited']:,} 个实体提及；其中 {graph['cross_passage_entities']:,} 个实体在多个片段出现。累计图构建 {graph['graph_build_seconds']:.1f} 秒。",
               "抽取后端为 spaCy en_core_web_sm 3.8.0：命名实体、名词短语、依存谓词；保留否定、情态、句内时间和原文出处。相同规范短语复用实体 ID，片段与实体多对多。关系区分 predicate 与 co_mention，全部标记为自动抽取、未经验证。图路径用于定位跨片段支持，不等于可信逻辑证明；仍有实体边界、指代和关系解析错误。","",
               "## 分组","",
               "完整分组机器结果在 summary.json。下表按子领域列出：","",
               "| 领域 | N | Direct | BM25 | v1 | v2 | v2 无图 |","|---|---:|---:|---:|---:|---:|---:|"]
    for domain in summaries['direct']['sub_domain']:
        cells=[f"{summaries[m]['sub_domain'][domain]['accuracy']:.1f}%" for m in ('direct','bm25','repar_v1','repar_v2','repar_tree_v2')]
        report.append(f"| {domain} | {summaries['direct']['sub_domain'][domain]['n']} | "+" | ".join(cells)+" |")
    report += ["","## 材料","",
               "- `artifacts/experiment_v2/*.jsonl`、`summary.json`、`per_question.csv`：全量结果。",
               "- `artifacts/traces_qwen08_v2/`：plan/action/observation/reflection 原始轨迹。",
               "- `artifacts/indexes_v2/`：实体图缓存与统计；`artifacts/structure_audit_v2.json`：全部出处审计。",
               "- `artifacts/implementation_manifest_v2.json`：本轮代码/配置；前期计时修正版本另存。",
               "- `webserver/`：可直接将 v1 与 v2 作 one-vs-one 对比，并读取实体、关系和 Trace。",""]
    Path('docs/EXPERIMENT_V2.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({m:summaries[m]['overall']['accuracy'] for m in conditions},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
