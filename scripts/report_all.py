"""Audit all 1500 predictions and report accuracy, cost and the MTP decision."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from repar.io import load_public_samples, read_json, read_jsonl, write_json
from report import METHODS, paired


VARIANTS = {
    "standard": ("artifacts/experiment", "configs/qwen08_standard.json", "标准 Q4"),
    "mtp_off": ("artifacts/experiment_mtp_off", "configs/qwen08_mtp_off.json", "MTP 权重，推测关闭"),
    "mtp_on": ("artifacts/experiment_mtp", "configs/qwen08_mtp.json", "MTP 权重，推测开启"),
}
NAMES = {"direct": "直接回答", "bm25": "BM25 RAG", "repar": "RePAR", "repar_tree": "RePAR 禁用图"}


def service(row):
    return sum(c.get("timings", {}).get("prompt_ms", 0) + c.get("timings", {}).get("predicted_ms", 0)
               for c in row["calls"])/1000


def speed_comparison(off, on, ids):
    x = np.array([service(off[i]) for i in ids])
    y = np.array([service(on[i]) for i in ids])
    draws = np.random.default_rng(42).integers(0, len(ids), size=(20000, len(ids)))
    ratios = x[draws].mean(axis=1)/y[draws].mean(axis=1)
    return {"service_speedup": float(x.mean()/y.mean()),
            "service_speedup_bootstrap_95": np.percentile(ratios, [2.5, 97.5]).tolist(),
            "service_reduction_percent": float((1-y.mean()/x.mean())*100),
            "off_mean_seconds": float(x.mean()), "on_mean_seconds": float(y.mean()),
            "client_wall_speedup": float(np.mean([off[i]["seconds"] for i in ids])/
                                           np.mean([on[i]["seconds"] for i in ids])),
            "off_cached_calls": sum(c["cached"] for i in ids for c in off[i]["calls"]),
            "on_cached_calls": sum(c["cached"] for i in ids for c in on[i]["calls"])}


def probe_stats(variant):
    p = read_json(f"artifacts/deployment/probe_{variant}.json")
    # Exclude the entire first repetition to remove cold-start graph allocations.
    rows = [r for r in p["rows"] if r["repeat"] > 0]
    predicted = sum(max(0, r["timings"].get("predicted_n", 0)-1) for r in rows)
    ms = sum(r["timings"].get("predicted_ms", 0) for r in rows)
    drafts = sum(r["timings"].get("draft_n", 0) for r in rows)
    accepted = sum(r["timings"].get("draft_n_accepted", 0) for r in rows)
    return {"calls": len(rows), "decode_tokens_per_second": predicted/(ms/1000),
            "tokens": predicted, "decode_seconds": ms/1000,
            "draft_tokens": drafts, "draft_tokens_accepted": accepted,
            "draft_acceptance_ratio": accepted/drafts if drafts else None,
            "same_output_between_repeats": sum(p["rows"][j]["response"] == p["rows"][j+4]["response"]
                                               for j in range(8)),
            "protocol": "4 fixed synthetic prompts x 3 repetitions; report repetitions 1 and 2; raw timings retained even for cached re-analysis"}


def main():
    ids = [r["_id"] for r in load_public_samples("data/longbench_v2.json")]
    for variant, (root, config, _) in VARIANTS.items():
        subprocess.run([sys.executable, "-X", "utf8", "scripts/report.py", "--root", root,
                        "--config", config, "--report", f"docs/REPORT_{variant.upper()}.md"], check=True)
        subprocess.run([sys.executable, "-X", "utf8", "scripts/plot_results.py", "--root", root], check=True)
    summaries = {v: read_json(Path(info[0]) / "summary.json") for v, info in VARIANTS.items()}
    rows = {v: {m: {r["_id"]: r for r in read_jsonl(Path(info[0]) / (m+".jsonl"))}
                for m in METHODS} for v, info in VARIANTS.items()}
    for v in rows:
        for m in METHODS:
            assert set(rows[v][m]) == set(ids)
            for i in ids:
                assert rows[v][m][i]["context_sha256"] == rows["standard"]["direct"][i]["context_sha256"]
    comparisons = {}
    for a, b in (("mtp_on", "mtp_off"), ("mtp_off", "standard"), ("mtp_on", "standard")):
        comparisons[a+"_vs_"+b] = {
            m: {**paired(rows[a][m], rows[b][m], ids),
                "changed_predictions": sum(rows[a][m][i]["pred"] != rows[b][m][i]["pred"] for i in ids),
                **speed_comparison(rows[b][m], rows[a][m], ids)} for m in METHODS}
    probes = {v: probe_stats(v) for v in VARIANTS}
    mtp = comparisons["mtp_on_vs_mtp_off"]
    accuracy_ok = all(comparisons[comparison][m]["difference_pp"] >= -.800001
                      for comparison in ("mtp_on_vs_mtp_off", "mtp_on_vs_standard")
                      for m in ("direct", "repar"))
    decode_gain = probes["mtp_on"]["decode_tokens_per_second"]/probes["mtp_off"]["decode_tokens_per_second"]
    decode_gain_standard = probes["mtp_on"]["decode_tokens_per_second"]/probes["standard"]["decode_tokens_per_second"]
    speed_ok = (all(comparisons[c]["repar"]["service_reduction_percent"] >= 20
                    for c in ("mtp_on_vs_mtp_off", "mtp_on_vs_standard"))
                and min(decode_gain, decode_gain_standard) >= 1.2)
    decision = {"enable_mtp": bool(accuracy_ok and speed_ok), "accuracy_operational_rule_met": bool(accuracy_ok),
                "speed_operational_rule_met": bool(speed_ok), "synthetic_decode_speedup": decode_gain,
                "synthetic_decode_speedup_vs_standard": decode_gain_standard,
                "repar_wall_speedup_vs_mtp_off": mtp["repar"]["client_wall_speedup"],
                "repar_wall_speedup_vs_standard": comparisons["mtp_on_vs_standard"]["repar"]["client_wall_speedup"],
                "criterion": read_json("artifacts/experiment_protocol.json")["mtp_default_criterion"],
                "note": "Accuracy rule is operational, not a statistical equivalence claim. Speed uses server computation, which is an end-to-end component rather than client wall time."}
    out = Path("artifacts/combined")
    out.mkdir(exist_ok=True)
    write_json(out/"summary.json", {"variants": summaries, "paired": comparisons,
                                    "probes": probes, "mtp_decision": decision})
    with (out/"conditions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["variant", "method", "n", "correct", "accuracy", "mean_prompt_tokens", "mean_completion_tokens",
                  "mean_prefill_seconds", "mean_decode_seconds", "mean_service_seconds", "mean_wall_seconds", "cached_calls"]
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        for v in VARIANTS:
            for m in METHODS:
                w.writerow({"variant": v, "method": m, **summaries[v]["methods"][m]["overall"]})
    with (out/"all_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["_id", "answer", *[v+"_"+m for v in VARIANTS for m in METHODS]])
        for i in ids:
            w.writerow([i, rows["standard"]["direct"][i]["answer"], *[rows[v][m][i]["pred"] for v in VARIANTS for m in METHODS]])

    acc = summaries["standard"]["methods"]["repar"]["overall"]["accuracy"]
    report = ["# RePAR 完整实验交付", "",
              f"全部 **125 道 Multi-Document QA × 4 种方法 × 3 种部署 = 1,500 个预测**已完成，未用部分样本代替全量结果。",
              f"标准 Q4 的 RePAR 准确率为 **{acc:.1f}%**，{'数值上超过' if acc > 44 else '未超过'}论文 Table 3 的 GPT-4o 历史参考值 44.0%。",
              f"MTP 默认决策：**{'启用' if decision['enable_mtp'] else '关闭'}**。同权重开启后的合成解码速度为关闭时的 **{decode_gain:.2f}×**，RePAR 服务计算时间变化为 **{-mtp['repar']['service_reduction_percent']:+.1f}%**。", "",
              "## 全量结果", "",
              "| 部署 | 方法 | 正确/125 | 准确率 | 平均预填充秒 | 平均解码秒 | 平均服务计算秒 | 平均实测墙钟秒 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for v, (_, _, name) in VARIANTS.items():
        for m in METHODS:
            s = summaries[v]["methods"][m]["overall"]
            report.append(f"| {name} | {NAMES[m]} | {s['correct']} | {s['accuracy']:.1f}% | {s['mean_prefill_seconds']:.3f} | {s['mean_decode_seconds']:.3f} | {s['mean_service_seconds']:.3f} | {s['mean_wall_seconds']:.3f} |")
    report += ["", "![全部条件的准确率与计算成本](../artifacts/combined/comparison.png)", "",
               "服务计算为每题所有 LLM 调用的 prompt_ms + predicted_ms 之和，覆盖预填充和解码。墙钟时间另含分词、索引加载、检索、HTTP 和本地缓存；禁用图条件可复用首次图调用前的相同 API 请求，墙钟值不能用来直接宣称其推理更快。服务计算与 tokens 对缓存调用仍计入首次真实推理成本。", "",
               "## MTP：同权重开关的配对比较", "",
               "两个 ModelScope 文件的 320 个共同张量中有 186 个不同，MTP 文件另有 15 个张量。因此标准文件与 MTP 文件的差异不能全部归于推测解码。此处仅比较同一份 MTP GGUF 的开关。",
               "| 方法 | 准确率差 pp | 配对 bootstrap 95% CI | 胜/负 | 预测变化题数 | 服务计算加速比 | 加速比 95% CI |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        s = mtp[m]
        report.append(f"| {NAMES[m]} | {s['difference_pp']:+.1f} | [{s['bootstrap_95'][0]:.1f}, {s['bootstrap_95'][1]:.1f}] | {s['wins']}/{s['losses']} | {s['changed_predictions']} | {s['service_speedup']:.2f}× | [{s['service_speedup_bootstrap_95'][0]:.2f}, {s['service_speedup_bootstrap_95'][1]:.2f}] |")
    report += ["", "配对重采样 20,000 次，seed=42；未作多重检验校正。相同 seed 不保证推测解码使用相同随机数路径，轨迹和预测可以变化。125 题不能证明严格的精度等价；置信区间宽度须结合差值看。RePAR 自适应轨迹可能不同，其服务时间比包含生成长度和取证路径变化。直接回答和 BM25 则保持输入文本一致。三种部署在同一 GPU 上按固定顺序运行，没有随机交错或多次全量重复；温度、动态频率和 Windows 桌面负载造成的系统性波动不会被题目 bootstrap 区间完全捕捉。", "",
               "| 短上下文合成吞吐探针 | 解码 tokens/s |", "|---|---:|"]
    for v, (_, _, name) in VARIANTS.items():
        report.append(f"| {name} | {probes[v]['decode_tokens_per_second']:.1f} |")
    report += ["", "探针为 4 个固定输入 × 3 次重复，输出上限 320 tokens，统计丢弃第 1 轮后的 8 次调用。吞吐沿用 llama.cpp 的解码计数口径：每次 predicted_n 减去计入预填充的首 token，再除以总解码时间；原始输入、生成全文和 timings 均保存。它只度量短输入的长输出，不能替代 benchmark 的实际耗时。", "",
               "| MTP 开启：真实 draft 统计 | 草稿 tokens | 接受 tokens | 接受率 |", "|---|---:|---:|---:|"]
    for m in METHODS:
        s = summaries["mtp_on"]["methods"][m]["overall"]
        ratio = s["draft_acceptance_ratio"]
        report.append(f"| {NAMES[m]} | {s['draft_tokens']} | {s['draft_tokens_accepted']} | {100*ratio:.1f}% |" if ratio is not None else f"| {NAMES[m]} | 0 | 0 | 无草稿 |")
    report += ["", "接受率采用 API 的 draft_n_accepted / draft_n；服务日志还保存每次调用的接受统计。b12500 的 /props 中 speculative.types 是请求默认值，即使通过启动参数启用 MTP 也显示 none；实际启用以启动参数、creating MTP draft context 日志及非零 draft_n / draft_n_accepted 三者交叉验证。这里只测试最多 6 个 draft tokens 的固定配置，没有按测试集成绩搜索草稿长度或采样设置；不能将本配置结果外推为所有 MTP 部署的速度。", "",
               "默认判据在汇总准确率前保存：相对标准 Q4 和同权重 MTP 关闭两个对照，direct 与 RePAR 均最多下降 1/125 题，RePAR 平均服务计算时间至少降低 20%，且合成解码吞吐至少提高 20%。这是工程选择规则，不是等价性检验。", "",
               "```json", json.dumps(decision, ensure_ascii=False, indent=2), "```", "",
               "## 核心结构与实际运行", "",
               "有序语义树保持连续原文范围，以段落/句子为优先边界；片段连接带字符出处的抽取式命题节点，图提供实体、概念、词汇关联及相邻片段桥接。tree_browse / tree_search / graph_search / read 都有可组合参数。控制器根据观察生成下一轮调用、反思和策略更新。算法细节见 [DESIGN.md](DESIGN.md)。",
               "当前语义表示使用 TF-IDF，知识抽象使用抽取式命题。图边是相关证据的检索假设，并未验证为因果、矛盾或逻辑蕴含。任务内策略修订没有改变模型权重，也不代表跨任务 RSI。", "",
               "| 部署 | 成功树查询 | 成功图查询 | 使用图的题 | ≥2 轮取证的题 | 重复调用错误 | 预算耗尽 | 回退题 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for v, (_, _, name) in VARIANTS.items():
        o = summaries[v]["operations"]["repar"]
        report.append(f"| {name} | {o['tool_counts'].get('tree_search',0)} | {o['tool_counts'].get('graph_search',0)} | {o['graph_used_samples']} | {o['two_retrieval_rounds_samples']} | {o['duplicate_call_errors']} | {o['termination'].get('budget',0)} | {o['fallbacks']} |")
    report += ["", "反思字段和策略文本改变不等于有效自进化；上述调用统计用于判断 0.8B 控制器是否真正利用了图和多轮取证。若图访问稀少，禁用图消融不能证明图结构无用，只能反映本控制器的实际使用。重复调用会被拒绝并反馈；finish 与工具调用冲突时先执行工具；至少两轮取证后允许主动结束，最多 5 轮，耗尽预算照实记录。当前原型没有证明 RePAR 的精度收益，也没有实现超越历史大模型的研究目标。更详细的证据与改进假设见 [失败分析](FAILURE_ANALYSIS.md)。", "",
               "## 数据、部署与公平性", "",
               "数据为 ModelScope LongBench-v2 的全部 125 道 Multi-Document QA；索引只接触 context，不接触问题、选项或答案。问题和选项只用于在线查询；答案仅由评测器打分。没有根据测试集准确率选择提示或检索参数。JSON 协议预检使用了前两道 benchmark 的无标签输入，依据格式和重复调用问题做可靠性修正后固定配置；这属于协议适配时接触过测试输入，不能称为完全未见测试输入的预注册实验。协议预检、旧 27B 运行均归档，未混入此次 1,500 个结果。",
               "三组都在同一张 RTX 3090 上顺序运行 llama.cpp b12500，Q4_K_M，262,144-token 窗口，parallel=1，seed=42，temperature=0.1，关闭 thinking。MTP 使用 draft-mtp、最多 6 个 draft tokens。当前模型不调用 9090 端口的原 27B 服务。",
               "模型来源：[标准 GGUF](https://www.modelscope.cn/models/unsloth/Qwen3.5-0.8B-GGUF)、[MTP GGUF](https://www.modelscope.cn/models/unsloth/Qwen3.5-0.8B-MTP-GGUF)；均选择 Q4_K_M。数据来源：[LongBench-v2](https://www.modelscope.cn/datasets/ZhipuAI/LongBench-v2)，历史对照：[论文 Table 3](https://arxiv.org/html/2412.15204v1#S4.T3)。下载的具体 revision、完整 SHA256 和每题上下文校验均保存在 artifacts 中。",
               f"直接回答有 {summaries['standard']['direct_full_context_n']} 题完整输入原文，{summaries['standard']['direct_truncated_n']} 题超过窗口，使用官方首尾截断规则。检索方法始终对完整原文构建索引。完整输入/截断子集的准确率在每部署详细报告中单列。",
               "最终答案统一使用 GBNF 约束为 The correct answer is (A/B/C/D)。这避免 0.8B 在预算内未输出选项，但属于官方自由生成设置的明确偏离。RePAR 额外控制器计算不与直接回答等预算；输入/输出 tokens 和调用数均公开。论文 GPT-4o 44.0% 为 2024 年历史参考，没有在本机复测，也不能据此声称优于所有当前大模型。", "",
               "## 原始材料与复现", "",
               "- [标准 Q4 详细报告](REPORT_STANDARD.md)、[MTP 关闭详细报告](REPORT_MTP_OFF.md)、[MTP 开启详细报告](REPORT_MTP_ON.md)：分领域/长度/难度、配对统计、置信区间、成本、操作审计。",
               "- `artifacts/combined/conditions.csv`：12 条件汇总；`all_predictions.csv`：逐题全部预测；`summary.json`：完整机器可读统计。",
               "- `artifacts/experiment*/*.jsonl`：原始评测记录；`artifacts/traces_qwen08_*/`：每轮模型输出、调用参数、原文证据、反思与结束原因。",
               "- `artifacts/cache/llm/`：内容寻址的原始请求/响应；`artifacts/deployment/`：模型版本、SHA256、张量比较、MTP 探针与 CUDA 日志；`artifacts/implementation_manifest.json`：推理代码/配置/提示校验。",
               "- `artifacts/structure_audit.json`：全部 125 个知识库的字符出处、树覆盖、图端点及查询延迟审计。", "",
               "```powershell", "conda activate RePAR", 'python -m pip install -e ".[dev]"', "python -m pytest -q",
               "python scripts/download_assets.py", "python scripts/build_indexes.py", "python scripts/run_experiments.py", "python scripts/finalize_results.py", "```", ""]
    Path("docs/EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    plot(summaries, out)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def plot(summaries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    labels = ["Direct", "BM25", "RePAR", "RePAR, graph disabled"]
    x = np.arange(4)
    for j, (v, label, color) in enumerate((("standard", "Standard Q4", "#657c96"),
                                          ("mtp_off", "MTP weights, spec off", "#55a895"),
                                          ("mtp_on", "MTP weights, spec on", "#d09b58"))):
        ss = [summaries[v]["methods"][m]["overall"] for m in METHODS]
        axes[0].bar(x+(j-1)*.25, [s["accuracy"] for s in ss], width=.24, label=label, color=color)
        axes[1].bar(x+(j-1)*.25, [s["mean_service_seconds"] for s in ss], width=.24, label=label, color=color)
    axes[0].axhline(44, color="#a15569", ls="--", lw=1, label="GPT-4o 44% (historical)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("LongBench v2 Multi-Doc QA: all 125 questions")
    axes[1].set_ylabel("Mean summed server computation (seconds/question)")
    axes[1].set_yscale("log")
    axes[1].set_title("Prefill + decode for all LLM calls")
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.savefig(out/"comparison.png", dpi=220)
    fig.savefig(out/"comparison.svg")


if __name__ == "__main__":
    main()
