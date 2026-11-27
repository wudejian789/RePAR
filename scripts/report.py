"""Strict full-coverage evaluation; never select a configuration by benchmark accuracy."""
from __future__ import annotations

import csv
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from repar.io import load_public_samples, read_json, read_jsonl, write_json


ROOT = Path("artifacts/experiment")
METHODS = ["direct", "bm25", "repar", "repar_tree"]
NAMES = {"direct": "Qwen3.5 0.8B 直接回答", "bm25": "Qwen3.5 0.8B + BM25 RAG",
         "repar": "Qwen3.5 0.8B + RePAR", "repar_tree": "RePAR 禁用图执行"}


def wilson(correct, n):
    if not n:
        return [None, None]
    z, p = 1.959963984540054, correct / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [100*(center-margin), 100*(center+margin)]


def stats(rows):
    n = len(rows)
    correct = sum(bool(x["judge"]) for x in rows)
    prompt = sum(sum(c["usage"].get("prompt_tokens", 0) for c in x["calls"]) for x in rows)
    completion = sum(sum(c["usage"].get("completion_tokens", 0) for c in x["calls"]) for x in rows)
    service = [sum((c.get("timings", {}).get("prompt_ms", 0) +
                    c.get("timings", {}).get("predicted_ms", 0))/1000 for c in x["calls"]) for x in rows]
    wall = [x["seconds"] for x in rows]
    prefill = sum(c.get("timings", {}).get("prompt_ms", 0) for x in rows for c in x["calls"])/1000
    decode = sum(c.get("timings", {}).get("predicted_ms", 0) for x in rows for c in x["calls"])/1000
    # llama.cpp charges the first generated token to prefill; its decode rate
    # uses (predicted_n - 1), which matters for our short constrained answers.
    predicted = sum(max(0, c.get("timings", {}).get("predicted_n", 0)-1) for x in rows for c in x["calls"])
    drafts = sum(c.get("timings", {}).get("draft_n", 0) for x in rows for c in x["calls"])
    accepted = sum(c.get("timings", {}).get("draft_n_accepted", 0) for x in rows for c in x["calls"])
    return {"n": n, "correct": correct, "accuracy": 100*correct/n if n else None,
            "wilson_95": wilson(correct, n), "invalid": sum(x["pred"] is None for x in rows),
            "prompt_tokens": prompt, "completion_tokens": completion,
            "mean_prompt_tokens": prompt/n if n else None, "mean_completion_tokens": completion/n if n else None,
            "total_llm_calls": sum(len(x["calls"]) for x in rows),
            "mean_prefill_seconds": prefill/n if n else None,
            "mean_decode_seconds": decode/n if n else None,
            "decode_tokens_per_second": predicted/decode if decode else None,
            "draft_tokens": drafts, "draft_tokens_accepted": accepted,
            "draft_acceptance_ratio": accepted/drafts if drafts else None,
            "mean_service_seconds": float(np.mean(service)) if service else None,
            "p95_service_seconds": float(np.percentile(service, 95)) if service else None,
            "mean_wall_seconds": float(np.mean(wall)) if wall else None,
            "p95_wall_seconds": float(np.percentile(wall, 95)) if wall else None,
            "cached_calls": sum(c["cached"] for x in rows for c in x["calls"]),
            "truncated_generations": sum(c.get("finish_reason") == "length" for x in rows for c in x["calls"])}


def paired(a, b, ids):
    x = np.array([int(a[i]["judge"]) - int(b[i]["judge"]) for i in ids])
    rng = np.random.default_rng(42)
    samples = x[rng.integers(0, len(ids), size=(20000, len(ids)))].mean(axis=1) * 100
    wins, losses = int((x > 0).sum()), int((x < 0).sum())
    return {"difference_pp": float(x.mean()*100), "bootstrap_95": np.percentile(samples, [2.5, 97.5]).tolist(),
            "wins": wins, "losses": losses, "mcnemar_exact_p": float(binomtest(wins, wins+losses, .5).pvalue)
            if wins+losses else 1.0, "bootstrap_samples": 20000, "seed": 42}


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/experiment")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--report", default="docs/EXPERIMENT_REPORT.md")
    args = parser.parse_args()
    ROOT = Path(args.root)
    config = read_json(args.config)
    data = load_public_samples("data/longbench_v2.json")
    ids = [x["_id"] for x in data]
    labels = {x["_id"]: x for x in data}
    methods, mapped = {}, {}
    for method in METHODS:
        rows = read_jsonl(ROOT / (method + ".jsonl"))
        actual = [r["_id"] for r in rows]
        assert len(actual) == len(set(actual)) == len(ids), f"Incomplete/duplicate condition {method}: {len(actual)}/{len(ids)}"
        assert set(actual) == set(ids), f"Sample mismatch in {method}"
        for row in rows:
            assert row["answer"] == labels[row["_id"]]["answer"]
            assert row["judge"] == (row["pred"] == row["answer"])
        mapped[method] = {r["_id"]: r for r in rows}
        rows = [mapped[method][i] for i in ids]
        methods[method] = {"overall": stats(rows)}
        for dimension in ("length", "difficulty", "sub_domain"):
            methods[method][dimension] = {value: stats([r for r in rows if r[dimension] == value])
                                          for value in sorted({r[dimension] for r in rows})}
    full_ids = [i for i in ids if not mapped["direct"][i]["truncated"]]
    truncated_ids = [i for i in ids if mapped["direct"][i]["truncated"]]
    for method in METHODS:
        methods[method]["direct_full_context_subset"] = stats([mapped[method][i] for i in full_ids])
        methods[method]["direct_truncated_subset"] = stats([mapped[method][i] for i in truncated_ids])
    comparisons = {f"repar_vs_{b}": paired(mapped["repar"], mapped[b], ids)
                   for b in ("direct", "bm25", "repar_tree")}
    operational = {}
    for method in ("repar", "repar_tree"):
        traces = [read_json(mapped[method][i]["trace_path"]) for i in ids]
        tools = Counter(c["name"] for t in traces for s in t["trace"]
                        for c in s.get("executed", []) if "name" in c)
        rounds = [mapped[method][i]["rounds"] for i in ids]
        operational[method] = {"tool_counts": dict(tools), "mean_rounds": float(np.mean(rounds)),
                               "round_distribution": dict(Counter(rounds)),
                               "fallbacks": sum(t["fallback"] for t in traces),
                               "termination": dict(Counter(t["termination"] for t in traces)),
                               "graph_used_samples": sum(any(c.get("name") == "graph_search"
                                   for s in t["trace"] for c in s.get("executed", [])) for t in traces),
                               "policy_revisions": sum(len(t["policies"]) >= 2 and t["policies"][0] != t["policies"][-1]
                                                       for t in traces),
                               "tool_errors": sum("error" in c for t in traces for s in t["trace"]
                                                  for c in s.get("executed", [])),
                               "duplicate_call_errors": sum("Repeated successful call" in c.get("error", "")
                                                  for t in traces for s in t["trace"] for c in s.get("executed", [])),
                               "finish_with_calls": sum(bool(s.get("decision", {}).get("finish")) and
                                                         bool(s.get("decision", {}).get("tool_calls"))
                                                         for t in traces for s in t["trace"]),
                               "two_retrieval_rounds_samples": sum(sum(any(c.get("observation", {}).get("results")
                                                   for c in s.get("executed", [])) for s in t["trace"]) >= 2 for t in traces),
                               "json_errors": sum("error" in s for t in traces for s in t["trace"])}
    unique_indexes = {mapped["repar"][i]["context_sha256"]: mapped["repar"][i]["index_stats"] for i in ids}
    indexing = {"unique_contexts": len(unique_indexes), "build_seconds": sum(x["build_seconds"] for x in unique_indexes.values()),
                "leaves": sum(x["leaves"] for x in unique_indexes.values()),
                "tree_nodes": sum(x["tree_nodes"] for x in unique_indexes.values()),
                "knowledge_nodes": sum(x["knowledge_nodes"] for x in unique_indexes.values()),
                "edges": sum(x["edges"] for x in unique_indexes.values())}
    summary = {"coverage": {m: len(mapped[m]) for m in METHODS}, "methods": methods,
               "comparisons": comparisons, "operations": operational, "indexing": indexing,
               "direct_full_context_n": len(full_ids), "direct_truncated_n": len(truncated_ids),
               "historical_table3": {"GPT-4o": 44.0, "GLM-4-Plus": 42.4, "Qwen2.5-72B": 35.2,
                                      "Llama-3.1-8B": 30.4, "GLM-4-9B": 27.2}}
    if Path("artifacts/structure_audit.json").exists():
        summary["structure_audit"] = read_json("artifacts/structure_audit.json")["summary"]
    write_json(ROOT / "summary.json", summary)
    with (ROOT / "per_question.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["_id", "sub_domain", "difficulty", "length", "answer", "direct_truncated"]
        fields += [m + suffix for m in METHODS for suffix in ("_pred", "_correct", "_seconds", "_prompt_tokens", "_completion_tokens")]
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for i in ids:
            row = {k: labels[i][k] for k in fields[:5]}
            row["direct_truncated"] = mapped["direct"][i]["truncated"]
            for m in METHODS:
                r = mapped[m][i]
                row.update({m+"_pred": r["pred"], m+"_correct": r["judge"], m+"_seconds": r["seconds"],
                            m+"_prompt_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in r["calls"]),
                            m+"_completion_tokens": sum(c["usage"].get("completion_tokens", 0) for c in r["calls"])})
            w.writerow(row)

    acc = methods["repar"]["overall"]["accuracy"]
    outcome = "达到" if acc > 44 else "未达到"
    report = ["# RePAR：LongBench v2 Multi-Doc QA 完整实验", "",
              f"125/125 题、四个预先固定条件均已完成。RePAR 准确率 **{acc:.1f}%**，{outcome}数值上超过论文 Table 3 GPT-4o 44.0% 的目标。",
              "这是与论文历史结果的数值比较；没有重新调用 GPT-4o，不能据此宣称优于所有当前大模型。", "",
              "## 设置", "",
              "数据来自 ModelScope ZhipuAI/LongBench-v2，503 题中全部 125 道 Multi-Document QA；SHA-256、每题 ID 和上下文校验见 `artifacts/dataset_manifest.json`。",
              "子领域：Governmental 23、Academic 50、Financial 15、Multi-news 23、Legal 14；长度：short 58、medium 44、long 23。",
              f"本地 RTX 3090 部署 {config['model']}，Q4_K-Medium，{config['context_window']:,}-token 服务窗口，接口 {config['base_url']}。所有控制、检索编排与作答均使用这个 0.8B 底座，不调用原 9090 服务或更大模型。部署参数、GGUF 校验和及 CUDA 日志见 `artifacts/deployment/`。",
              "temperature=0.1、seed=42、enable_thinking=false；直接回答和最终答案最多 128 输出 tokens。所有条件的最终答案采用相同 GBNF 约束，只能输出 The correct answer is (A/B/C/D) 的合法句式，防止模型先解释导致 128-token 预算内未作答。该格式约束是对官方评测的明确偏离，不能称作逐字复现官方解码设置。RePAR 每轮控制器最多 768 tokens、最多 5 轮、每轮最多 3 个工具调用，证据记忆最多 10,000 words。",
              "不进行参数训练；本项目 Parametric Retrieval 指查询 API 的参数化。索引完全由 context 构建，不接触 question、choices、answer；查询可使用问题与四个选项。",
              "自进化指单题内查询策略、计划和证据记忆的反馈修正；没有模型权重自修改或跨题使用答案的训练。代码与配置在汇总打分前固定，见 `artifacts/implementation_manifest.json`。", "",
              "## 主结果", "",
              "| 方法 | 正确/总数 | 准确率 | Wilson 95% CI | 无效答案 | 平均 LLM 调用 |",
              "|---|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        s = methods[m]["overall"]
        report.append(f"| {NAMES[m]} | {s['correct']}/{s['n']} | {s['accuracy']:.1f}% | [{s['wilson_95'][0]:.1f}, {s['wilson_95'][1]:.1f}] | {s['invalid']} | {s['total_llm_calls']/s['n']:.2f} |")
    report += ["", "## 配对比较", "",
               "| RePAR 对照 | 准确率差（百分点） | 配对 bootstrap 95% CI | 胜/负题数 | 双侧精确 McNemar p |",
               "|---|---:|---:|---:|---:|"]
    for b in ("direct", "bm25", "repar_tree"):
        s = comparisons["repar_vs_" + b]
        report.append(f"| {NAMES[b]} | {s['difference_pp']:+.1f} | [{s['bootstrap_95'][0]:.1f}, {s['bootstrap_95'][1]:.1f}] | {s['wins']}/{s['losses']} | {s['mcnemar_exact_p']:.4f} |")
    report += ["", "Bootstrap 使用 20,000 次同题配对重采样、seed=42。统计检验未作多重比较校正。置信区间跨 0 时不能认定稳定提升。", "",
               "## 全上下文与截断", "",
               f"直接回答中 **{len(full_ids)}** 题完整装入上下文；**{len(truncated_ids)}** 题超出窗口，按官方实现保留 prompt 首尾各一半。输入预算 {config['direct_input_budget']:,} tokens，为聊天模板和 128-token 输出预留空间。",
               "主表是官方风格的可用窗口基线，并非所有题都完整输入了原文。RePAR 建索引时始终处理完整原文。", "",
               "| 方法 | 基线能完整输入的题 | 基线需要截断的题 |", "|---|---:|---:|"]
    for m in METHODS:
        a, b = methods[m]["direct_full_context_subset"], methods[m]["direct_truncated_subset"]
        report.append(f"| {NAMES[m]} | {a['correct']}/{a['n']} ({a['accuracy']:.1f}%) | {b['correct']}/{b['n']} ({b['accuracy']:.1f}%) |")
    for dimension, title in (("length", "长度"), ("difficulty", "难度"), ("sub_domain", "子领域")):
        report += ["", "## 按" + title + "拆分", "",
                   "| 分组 | N | 直接回答 | BM25 | RePAR | 无图 RePAR |", "|---|---:|---:|---:|---:|---:|"]
        for value in methods["direct"][dimension]:
            cells = [f"{methods[m][dimension][value]['accuracy']:.1f}%" for m in METHODS]
            report.append(f"| {value} | {methods['direct'][dimension][value]['n']} | " + " | ".join(cells) + " |")
    report += ["", "## 成本与运行情况", "",
               "| 方法 | 平均输入 tokens | 平均输出 tokens | 平均服务计算秒数 | P95 服务计算秒数 | 平均实测墙钟秒数 |",
               "|---|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        s = methods[m]["overall"]
        report.append(f"| {NAMES[m]} | {s['mean_prompt_tokens']:.0f} | {s['mean_completion_tokens']:.0f} | {s['mean_service_seconds']:.2f} | {s['p95_service_seconds']:.2f} | {s['mean_wall_seconds']:.2f} |")
    report += ["", "服务计算时间为 API 返回 prompt_ms + predicted_ms 的各次调用之和；墙钟时间包含排队、网络、API 请求缓存读取和本地工具执行。调用缓存命中也按原始实际 API token 和服务时间计入；因此复用已有相同请求不等于零推理成本。",
               f"完整索引共 {indexing['unique_contexts']} 个唯一上下文，{indexing['leaves']:,} 个叶片、{indexing['tree_nodes']:,} 个树节点、{indexing['edges']:,} 条图边，累计首次构建 {indexing['build_seconds']:.2f} 秒；该离线成本单独列出。", "",
               "```json", json.dumps(operational, ensure_ascii=False, indent=2), "```", "",
               "## 实现范围与解释边界", "",
               "树采用保持原文顺序的连续区间分割，在近似均衡的候选边界中优先切断 TF-IDF 邻接相似度较低的位置；叶片约 240 words，优先保留段落/句子边界。TF-IDF 是稀疏词汇语义代理，不是训练过的神经语义编码器。",
               "知识节点保留片段中显著的三个原文命题、概念词及因果/对比/时间线索；图边为共享实体、共享概念、词汇关联、原文相邻。它是一种带出处的软知识图，边本身不证明因果、蕴含或矛盾；最终判断依靠原文。",
               "树查询执行层次 beam 路由，hybrid 模式允许两个倒排检索候选补充召回；图查询允许种子、跳数、关系过滤和返回数量。禁用图执行的消融保留相同控制器、工具描述和预算，在 dispatcher 处阻断图调用并将错误反馈给控制器。首次图访问以前的状态完全一致，可复用相同请求缓存；未尝试图访问的题两组轨迹可完全一致。这衡量图访问能力被阻断的影响，不是重新设计无图控制器的成绩。",
               "控制器以 JSON 工具协议输出参数化调用，本地 dispatcher 执行；执行结果、反思、策略更新和终止原因逐轮保存。这不是伪造的固定查询轨迹，也不依赖服务端原生 function-calling 解析。",
               "125 题上的一次运行不能证明通用提升，四位量化、tokenizer 和 2026 年的底座也与 2024 年论文设置不同。论文的历史结果只作同任务参考；不能把框架增益与底座升级的贡献混在一起。", "",
               "## 复现与原始材料", "",
               "```powershell", "conda activate RePAR", "python -m pip install -e .", "python -m pytest -q",
               "python scripts/prepare_data.py", f"python scripts/run_suite.py --config {args.config} --output {ROOT}", "```", "",
               f"逐题对照：`{ROOT}/per_question.csv`；原始预测：同目录四个 `.jsonl`；完整 API 请求/响应：`artifacts/cache/llm`；每题 ReAct 轨迹：`{config['trace_dir']}`；汇总机器数据：`{ROOT}/summary.json`。",
               "论文：https://arxiv.org/html/2412.15204v1#S4.T3；官方评测：https://github.com/THUDM/LongBench/blob/main/pred.py；数据：https://www.modelscope.cn/datasets/ZhipuAI/LongBench-v2。", ""]
    Path(args.report).write_text("\n".join(report), encoding="utf-8")
    print("Complete report:", Path(args.report).resolve())
    for m in METHODS:
        print(m, methods[m]["overall"]["correct"], "/", len(ids), methods[m]["overall"]["accuracy"])


if __name__ == "__main__":
    main()
