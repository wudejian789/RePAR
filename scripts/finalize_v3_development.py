"""Finish frozen full conditions and a predeclared global review ablation.

Never edits predictions/grades. Copies only missing request-cache records with
explicit provenance; all generation/grade phases still execute their adapters.
"""
from datetime import datetime
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from repar.evaluation import extract_answer


ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_import(source, target, label):
    manifest = target/(label+".json")
    if manifest.exists(): return
    records = []
    for p in (source/"cache").rglob("*.json"):
        relative = p.relative_to(source/"cache")
        q = target/"cache"/relative
        q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists():
            # Preserve actual local calls even if their timings differ.
            if read(q)["request"] != read(p)["request"]:
                raise ValueError("Cache key collision with different real requests")
            status = "existing_preserved"
        else:
            shutil.copyfile(p, q); status = "copied"
        records.append({"source": str(p.relative_to(ROOT)), "target": str(q.relative_to(ROOT)),
                        "source_sha256": sha(p), "target_sha256": sha(q), "status": status})
    write(manifest, {"time": datetime.now().astimezone().isoformat(), "rule": "Exact real request only; never overwrite existing responses or copy predictions/grades", "records": records})


def run(command, logfile):
    with logfile.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def call_resources(calls, stage_key=lambda call: call["tag"]):
    """Separate logical request usage from actual uncached generation usage."""
    stages = {}
    for call in calls:
        stage = stages.setdefault(stage_key(call), {"requests": 0, "cache_hits": 0,
            "logical_prompt_tokens": 0, "logical_completion_tokens": 0,
            "uncached_prompt_tokens": 0, "uncached_completion_tokens": 0})
        usage = call["response"].get("usage", {})
        stage["requests"] += 1
        stage["cache_hits"] += bool(call["cached"])
        for kind in ("prompt", "completion"):
            tokens = usage.get(kind+"_tokens", 0)
            stage["logical_"+kind+"_tokens"] += tokens
            if not call["cached"]: stage["uncached_"+kind+"_tokens"] += tokens
    return stages


def wilson_interval(successes, total):
    """Descriptive binomial interval; not a held-out development guarantee."""
    z = 1.959963984540054
    p, scale = successes/total, 1+z*z/total
    center = (p+z*z/(2*total))/scale
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/scale
    return [center-radius, center+radius]


def main():
    paths = {i: ROOT/f"artifacts/repar_v3_20261119_v{i}" for i in range(1, 5)}
    while not (paths[2]/"summary.json").exists() or not (paths[3]/"math/grades.jsonl").exists() or len((paths[3]/"math/grades.jsonl").read_text(encoding="utf-8").splitlines()) != 19:
        time.sleep(10)
    print("All condition2 responses complete; condition3 math complete", flush=True)
    cache_import(paths[2], paths[3], "cache_import_after_condition2")
    run([sys.executable, "-X", "utf8", "scripts/run_repar_v3.py", "--config", "configs/v3_experiment_v3.json", "--phase", "longbench"], paths[3]/"longbench_run.log")
    run([sys.executable, "-X", "utf8", "scripts/run_repar_v3.py", "--config", "configs/v3_experiment_v3.json", "--phase", "report"], paths[3]/"report_run.log")
    print("Condition3 full completed", flush=True)
    paths[4].mkdir(parents=True, exist_ok=True)
    cache_import(paths[3], paths[4], "cache_import_from_review_condition")
    run([sys.executable, "-X", "utf8", "scripts/run_repar_v3.py", "--config", "configs/v3_experiment_v4.json", "--phase", "all"], paths[4]/"full_run.log")
    print("Global no-review ablation full completed", flush=True)
    for i in range(1, 5):
        run([sys.executable, "-X", "utf8", "artifacts/maintenance/audit_repar_v3.py", "--experiment", str(paths[i].relative_to(ROOT))], paths[i]/"audit.log")
        print("AUDITED", i, flush=True)
    report(paths)


def report(paths):
    experiments = []
    for i, path in paths.items():
        s, a = read(path/"summary.json"), read(path/"audit.json")
        if not a["passed"]: raise ValueError("Condition did not pass audit")
        predictions = {name: [json.loads(line) for line in (path/name/"predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                       for name in ("math", "longbench")}
        diagnostics = {}
        for name, values in predictions.items():
            diagnostics[name] = {"termination": dict(Counter(r.get("termination", "missing") for r in values)),
                "tool_kinds": dict(Counter(e["name"] for r in values for t in r["trace"] for e in t.get("executed", []))),
                "invalid_tool_requests": sum(e["observation"].get("status") == "invalid" for r in values for t in r["trace"] for e in t.get("executed", [])),
                "no_observed_evidence": sum(not r.get("evidence_ids") for r in values),
                "explicit_abstentions": sum(r["response"] == "Insufficient source evidence." for r in values),
                "failed_optional_reviews": sum(r.get("review", {}).get("status") == "format_failed" for r in values if r.get("review")),
                "review_revised_answers": sum(r.get("review", {}).get("status") == "issue" for r in values if r.get("review")),
                "partial_outputs_recovered": sum(r.get("output_checks", {}).get("partial_output_recovered", False) for r in values)}
            diagnostics[name]["solver_resources_by_stage"] = call_resources(
                [c for r in values for c in r["calls"]])
            diagnostics[name]["prediction_errors"] = dict(Counter(
                r.get("error", "unknown") for r in values if r["status"] != "ok"))
            if name == "longbench":
                drafts = []
                for r in values:
                    call = next((c for c in r["calls"] if c["tag"] == "v3_longbench_final"), None)
                    if call is not None:
                        choice = call["response"]["choices"][0]["message"]["content"]
                        drafts.append((r, extract_answer(choice)))
                diagnostics[name]["raw_initial_answers_posthoc"] = {
                    "available": len(drafts),
                    "correct": sum(draft == r["gold"] for r, draft in drafts),
                    "correct_in_error_rows": sum(
                        draft == r["gold"] and r["status"] != "ok" and r["correct"] for r, draft in drafts),
                    "correct_lost_in_error_rows": sum(
                        draft == r["gold"] and r["status"] != "ok" and not r["correct"] for r, draft in drafts),
                }
                reviewed = [r for r in values if r.get("review") and r["status"] == "ok"]
                changes = Counter()
                for r in reviewed:
                    before = extract_answer(r["draft_response"])
                    after = r["pred"]
                    if before != after:
                        changes["changed"] += 1
                        changes["wrong_to_correct"] += before != r["gold"] and after == r["gold"]
                        changes["correct_to_wrong"] += before == r["gold"] and after != r["gold"]
                        changes["wrong_to_other_wrong"] += before != r["gold"] and after != r["gold"]
                    else: changes["answer_unchanged"] += 1
                diagnostics[name]["review_answer_changes_posthoc"] = {
                    key: changes[key] for key in ("changed", "wrong_to_correct", "correct_to_wrong",
                                                   "wrong_to_other_wrong", "answer_unchanged")}
        math_grades = [json.loads(line) for line in (path/"math/grades.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        diagnostics["math"]["judge_resources_by_stage"] = call_resources(
            [c for r in math_grades for c in r.get("judge_calls", [])],
            stage_key=lambda c: "judge_repair" if c["tag"].endswith("_repair") else "judge")
        cache_imports = {p.name: dict(Counter(r.get("status", "copied") for r in read(p)["records"]))
                         for p in path.glob("cache_import*.json")}
        left, right = len(s["longbench"]["new_only_correct"]), len(s["longbench"]["old_only_correct"])
        discordant = left+right
        paired_p = min(1, 2*sum(math.comb(discordant,k) for k in range(min(left,right)+1))/2**discordant) if discordant else 1
        experiments.append({"condition": i, "summary": s, "audit_assertions": a["checks_total"],
                            "diagnostics": diagnostics, "longbench_paired_exact_two_sided_p_vs_v2": paired_p,
                            "longbench_wilson_95_interval": wilson_interval(s["longbench"]["correct"], 125),
                            "cache_imports": cache_imports,
                            "goal_check": {"math_exceeds_historical_bm25": s["math"]["score"] > 25,
                                           "longbench_not_lower_than_v2": s["longbench"]["correct"] >= 34}})
    result = {"completed_at": datetime.now().astimezone().isoformat(), "conditions": experiments,
              "exposure": "Post-hoc development; all conditions fully disclosed, not unseen confirmation",
              "source_policy": "All data, source partitions, tree and entity indexes unchanged",
              "reference": {"direct": {"math_score": 18, "math_full": 3, "longbench_correct": 26},
                            "bm25": {"math_score": 25, "math_full": 5, "longbench_correct": 29},
                            "legacy_v2": {"math_score": 14.5, "math_full": 2, "longbench_correct": 34}},
              "no_case_selection": "No per-question mixing or score-driven reruns; review enabled/disabled globally in separately frozen conditions"}
    target = ROOT/"artifacts/repar_v3_development_20261119"
    target.mkdir(parents=True, exist_ok=True)
    write(target/"summary.json", result)
    lines = ["# RePAR v3完整开发实验汇总", "", "原始教材、连续正文、片段划分、树与实体图均保持不变。作答底座仍为0.8B标准Q4、关闭thinking，27B只用于数学judge。", "",
             "| 条件 | MATH实验分/150 | MATH整题全对 | LongBench Multi-Document QA | 状态 |", "|---|---:|---:|---:|---|",
             "| 历史direct | 18 | 3/19 | 26/125（20.8%） | 历史参考 |", "| 历史BM25 | 25 | 5/19 | 29/125（23.2%） | 历史参考 |",
             "| 历史RePAR v2 | 14.5 | 2/19 | 34/125（27.2%） | 历史参考 |"]
    for entry in experiments:
        s = entry["summary"]
        lines.append(f"| v3条件{entry['condition']} | {s['math']['score']:.3f} | {s['math']['fully_correct']}/19 | {s['longbench']['correct']}/125（{100*s['longbench']['accuracy']:.1f}%） | 完整，{entry['audit_assertions']}项审计 |")
    lines += ["", "## 条件含义", "",
              "1. 初版通用执行、token证据记忆、可直接draft、有界计算及结构输出。完整负结果保留。",
              "2. 公共来源必需/参考契约、原任务查询锚点、中英文说明、显式输出标签语义、最多一次同模型复核。",
              "3. 通用输出失败恢复：主要答案字段先于说明；只有已完整生成的显式答案可从截断中恢复；可选复核失败保留有效初稿。其他契约不变。",
              "4. 预先固定的全局关闭可选复核消融。两套数据都关闭，不按某题/数据集的gold挑选是否复核。完全相同请求允许复用缓存，未复制预测/评分行。", "",
              "数学分为原rubric实验折算分，不是官方高考得分。新增JSON输出契约、计算和复核的资源/格式效应与检索效应没有被完全隔离，不能将所有变化归为检索收益。",
              "两套数据及旧结果此前已被查看；修复依据是通用契约问题、独立fixture和真实失败，不称未见测试确认，不保证绝对最高结果。", "",
              "## 验证与限制", "",
              "最终通用代码tests及webserver/tests共60项通过，3项上游警告；v3独立组件13项通过。源文件和模型调用审计逐条件留存。独立在线fixture仍有同模型反转比较/标签的已知失败，不把可解析或复核完成当作事实正确。",
              "默认旧v2入口保留；新v3为显式入口。只有达到目标的实测证据才能声称提高，否则完整报告未达到。不按MATH/LongBench在题级切换历史v2与v3。",
              "共享硬件、缓存和不同调用数单列，缓存墙钟不能宣称实测算法加速。中文图/多语dense/新抽取后端未实现，原始噪声保留。", ""]
    lines += ["## 完整性与目标检查", "", "| 条件 | MATH高于BM25的25分 | LongBench不低于旧v2的34题 | LongBench 95% Wilson区间 | LongBench相对v2配对p（描述性） |", "|---|---|---|---:|---:|"]
    for entry in experiments:
        g = entry["goal_check"]
        lo, hi = entry["longbench_wilson_95_interval"]
        lines.append(f"| {entry['condition']} | {g['math_exceeds_historical_bm25']} | {g['longbench_not_lower_than_v2']} | {lo:.1%}–{hi:.1%} | {entry['longbench_paired_exact_two_sided_p_vs_v2']:.4f} |")
    lines += ["", "同一题的已曝光开发比较和单次结果不足以证实稳定优势；配对检验仅描述性补充，不用来反选参数。没有逐题组合或按数据集静默切换不同条件。", "",
              "## 调用资源与运行状态", "",
              "| 条件 | 数据集 | 作答调用 | 缓存命中 | 输出格式修复 | 复核及修复调用 | 预测失败 | 未观察到来源 | 明示弃答 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in experiments:
        for dataset, d in entry["diagnostics"].items():
            s = entry["summary"][dataset]
            stages = d["solver_resources_by_stage"]
            reviews = sum(r["requests"] for tag, r in stages.items() if tag.startswith("v3_answer_review"))
            lines.append(f"| {entry['condition']} | {dataset} | {s['calls']} | {s['cache_hits']} | {s['repairs']} | {reviews} | {s['errors']} | {d['no_observed_evidence']} | {d['explicit_abstentions']} |")
    lines += ["", "各阶段逻辑请求和未缓存 token、数学评阅调用、工具类型、失败恢复和缓存导入计数见同目录 `summary.json`。逻辑 token 包含缓存响应记载的原始用量，不能当作本次实际推理消耗。", ""]
    lines += ["## LongBench 初稿与复核诊断", "",
              "| 条件 | 初稿正确 | 异常行中仍计分的正确初稿 | 复核由错改对 | 复核由对改错 | 正常返回初稿的复核失败 |",
              "|---|---:|---:|---:|---:|---:|"]
    for entry in experiments:
        d = entry["diagnostics"]["longbench"]
        drafts, edits = d["raw_initial_answers_posthoc"], d["review_answer_changes_posthoc"]
        lines.append(f"| {entry['condition']} | {drafts['correct']} | {drafts['correct_in_error_rows']} | {edits['wrong_to_correct']} | {edits['correct_to_wrong']} | {d['failed_optional_reviews']} |")
    lines += ["", "初稿诊断只解析已经记录的原始输出。条件2的LongBench适配器在复核前就写入初稿，复核异常只改变状态，初稿依然用于计分；条件3把同样的结果明确记录为保留初稿。MATH的旧适配器在最终接口整体返回后才写入答案，其失败行为不同。", ""]
    (target/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
