"""Audit and report the three-seed r31 evaluation."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import jsonschema

from repar.benchmarks.math_suite import usage
from repar.io import load_public_samples, read_json, write_json
from repar.v4 import objective_state
from report_benchmark_recommended import audit_condition
from report_math_r28 import audit_grade, saved_request, tree_query_counts
from run_benchmark_recommended import OLD, sha
from run_benchmark_r31 import CONFIG, DATASETS, MODEL, SEEDS, configuration, implementation_identity
from run_math_r28 import questions_and_judge
from run_math_r30 import configuration as r30_configuration
from repeat_benchmark_recommended import configuration as reference_configuration


SECTION = "## RePAR v4-r31 0.8B 目标进度与证据缺口"
PROMPTS = {"plan": objective_state.PLAN_PROMPT, "action": objective_state.ACTION_PROMPT,
           "reflection": objective_state.REFLECTION_PROMPT}


def audit_trace(folder, row, settings):
    stats = Counter(tree_query_counts(row["trace"]))
    requests = {}
    for call in row["calls"][:-1]:
        _, stage, number, *repair = call["tag"].split("_")
        record = read_json(folder / "cache/solver" / f"{call['key']}.json")
        assert record["response"] == call["response"]
        request = record["request"]
        assert request["messages"][0]["content"] == PROMPTS[stage]
        if stage in {"plan", "action"}:
            assert request["max_tokens"] == settings[f"{stage}_max_tokens"]
        requests[stage, int(number)] = record
        stats[f"{stage}_requests"] += 1
        stats["repairs"] += bool(repair)

    def read_control(stage, number):
        record = requests.pop((stage, number))
        request = record["request"]
        schema = request["response_format"]["json_schema"]["schema"]
        payload = json.loads(request["messages"][1]["content"])
        value = json.loads(record["response"]["choices"][0]["message"]["content"])
        jsonschema.validate(value, schema)
        return value, payload, schema

    goals = objective_state.ObjectiveState()
    plan_requested = True
    completed = set()
    from repar.v4.knowledge import request_identity

    for step in row["trace"]:
        number = step["round"]
        assert step["protocol"] == "objective_state_v4" and step["planned"] == plan_requested
        if plan_requested:
            plan, payload, schema = read_control("plan", number)
            assert plan == step["plan_proposal"]
            assert payload["objective_state"] == goals.snapshot()
            assert schema == goals.plan_request(payload)[1]
            stats["replans"] += number > 1
            goals.apply_plan(plan)
            stats["multi_objective_plans"] += len(goals.active) > 1

        assert goals.snapshot() == step["objective_state_before"]
        action, payload, schema = read_control("action", number)
        pending = goals.pending()
        assert payload["objective_state"] == goals.snapshot() and payload["pending_objectives"] == pending
        assert action == step["action_proposal"]
        for branch in schema["oneOf"]:
            if branch["properties"]["tool"]["const"] != "answer":
                assert branch["properties"]["objective_id"]["enum"] == pending
        stats["action_proposals"] += 1
        if "blocked_before_execution" in step:
            block = step["blocked_before_execution"]
            assert block["proposal"] == action and not step["executed"]
            stats["blocked_actions"] += 1
            stats["duplicate_actions"] += block["reason"] == "Duplicate completed action"
            continue

        decision = step["decision"]
        assert decision["tool"] == action["tool"]
        assert decision.get("objective_id") == action.get("objective_id")
        action_parameters = dict(action["parameters"])
        if "query" in action_parameters:
            action_parameters["query"] = action_parameters["query"].strip() or payload["task"]
        assert all(decision["parameters"][key] == value for key, value in action_parameters.items())
        if action["tool"] == "tree_search" and action_parameters["query"] != payload["task"]:
            assert decision["parameters"]["anchor_query"] == payload["task"]
        if action["tool"] == "answer":
            assert not step["executed"]
            stats["direct_answer_choices"] += 1
            continue

        execution, = step["executed"]
        oid = action["objective_id"]
        assert oid in pending and decision["gap"] == goals.records[oid]["gap"]
        assert execution["name"] == decision["tool"] and execution["parameters"] == decision["parameters"]
        identity = request_identity(execution["name"], execution["parameters"])
        assert identity not in completed
        completed.add(identity)
        stats["tool_executions"] += 1
        stats[f"tool_{execution['name']}"] += 1
        stats["invalid_tool_executions"] += execution["observation"]["status"] == "invalid"

        reflection, payload, _ = read_control("reflection", number)
        assert reflection == step["reflection"] and payload["objective_state"] == goals.snapshot()
        assert payload["addressed_gap"] == decision["gap"]
        assert payload["observation"] == execution["visible_observation"]
        assert payload["available_sources"] == step["reflection_context"]["available_sources"]
        for update in reflection["updates"]:
            assert set(update.get("sources", [])) <= set(payload["available_sources"])
        rejected = goals.apply_reflection(reflection)
        assert rejected == step.get("rejected_objective_updates", [])
        assert goals.snapshot() == step["objective_state_after"]
        stats["rejected_updates"] += len(rejected)
        plan_requested = reflection["next"] == "replan"
        closed = not goals.pending() and reflection["next"] == "continue"
        assert closed == (step.get("control_transition") == "all_active_objectives_covered")
        stats["closed_after_continue"] += closed
    assert not requests
    assert row["state"]["objective_state"] == goals.snapshot()
    assert set(row["state"]["completed_actions"]) == completed
    stats["finish_" + row["termination"]] += 1
    stats["single_tool_cases"] += len(completed) == 1
    return stats


def audit_run(settings, dataset, questions, judge, deployment):
    root = Path(settings["output"])
    seed = settings["answer_seed"]
    folder = root / MODEL / dataset / "repar"
    is_math = dataset == "math"
    reference = r30_configuration(seed) if is_math else reference_configuration(seed)
    prior = Path(reference["output"]) / MODEL / dataset / "repar"
    rows = {p.stem: read_json(p) for p in (folder / "cases").glob("*.json")}
    assert rows.keys() == questions.keys()
    grades = {}
    if is_math:
        grades = {p.stem: read_json(p) for p in (folder / "grades").glob("*.json")}
        assert grades.keys() == questions.keys()
        manifest = read_json(root / "08_judge_protocol.json")
        assert manifest["deployment"] == deployment
        assert all(manifest["config"][key] == value for key, value in judge.items() if key != "deployment_id")
        for path, expected in manifest["files"].items():
            assert sha(Path(path)) == expected, path
    result = audit_condition(settings, root, MODEL, dataset, "repar", questions, judge, not is_math)
    controls, pairs, judge_calls = Counter(), {}, []
    for qid, row in rows.items():
        prior_row = read_json(prior / "cases" / f"{qid}.json")
        assert row["status"] == "ok" and row["agent_version"] == settings["agent_version"]
        assert row["query_tokenizer"] == ("chinese" if is_math else "historical")
        assert all(row[key] == prior_row[key] for key in ("task", "context_sha256", "base_url"))
        assert row["index_reuse"]["path"] == prior_row["index_reuse"]["path"]
        if is_math:
            assert row["index_reuse"]["sha256"] == prior_row["index_reuse"]["sha256"]
        body = saved_request(folder, row)
        assert body == {**saved_request(prior, prior_row), "messages": body["messages"]}
        stats = audit_trace(folder, row, settings)
        controls.update(stats)
        score, prior_score = row.get("correct"), prior_row.get("correct")
        if is_math:
            grade = grades[qid]
            audit_grade(folder / "cache/judge", grade, questions[qid], row["response"], judge)
            judge_calls.extend(grade["judge_calls"])
            score = grade["score"]
            prior_score = read_json(prior / "grades" / f"{qid}.json")["score"]
        pairs[qid] = {"score": score, "prior_score": prior_score, "controls": dict(stats),
                      "finish": row["calls"][-1]["response"]["choices"][0]["finish_reason"]}
    result.update(complete=True, grade_count=len(grades), score=sum(p["score"] for p in pairs.values()),
                  prior_version="r30" if is_math else "r27",
                  prior_score=sum(p["prior_score"] for p in pairs.values()), controls=dict(controls),
                  judge_usage=usage(judge_calls), comparisons=pairs)
    if is_math:
        result["full"] = sum(g["fully_correct"] for g in grades.values())
    return result


def publish(runs, root):
    lines = [SECTION, "",
             "0.8B 作答，种子为 42、43、44。MATH 使用中文索引、温度 0.1、自由文本 4K 上限和 27B 温度 0 匿名阅卷。"
             "LongBench v2 Multi-Document 使用历史索引、温度 0.1、128-token 上限与四选一 GBNF。", "",
             "| 评测 | 对照 | 对照三轮 | r31 三轮 | 对照均值 | r31 均值 |",
             "| --- | --- | --- | --- | ---: | ---: |"]
    for dataset, label, denominator in [("math", "MATH", 150), ("longbench", "LongBench v2 Multi-Document", 125)]:
        rows = [row[dataset] for row in runs.values()]
        previous = [row["prior_score"] for row in rows]
        current = [row["score"] for row in rows]
        fmt = ".3f" if dataset == "math" else ".0f"
        old_scores = "、".join(format(value, fmt) for value in previous)
        new_scores = "、".join(format(value, fmt) for value in current)
        lines.append(f"| {label} | {rows[0]['prior_version']} | {old_scores} | {new_scores} | "
                     f"{mean(previous):.3f}/{denominator} | {mean(current):.3f}/{denominator} |")
    lines.extend(["", "| 评测 | r31 秒/题 | r31 输入 token/轮 | r31 输出 token/轮 | r31 作答与控制请求 |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for dataset in DATASETS:
        rows = [row[dataset] for row in runs.values()]
        lines.append(f"| {dataset} | {mean(row['case_mean'] for row in rows):.2f} | "
                     f"{mean(row['costs']['input']['sum'] for row in rows):,.0f} | "
                     f"{mean(row['costs']['output']['sum'] for row in rows):,.0f} | "
                     f"{sum(row['calls'] for row in rows)} |")
    lines.extend(["", f"[逐题评分、耗时与控制状态审计]({(root / 'summary.json').as_posix()})"])
    report = Path("all_results_new.md")
    body = report.read_text(encoding="utf-8").split(SECTION, 1)[0].rstrip()
    temp = report.with_suffix(".md.tmp")
    temp.write_text(body + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(report)


def main():
    math, judge, deployment = questions_and_judge()
    data_path = read_json(OLD[MODEL] / "protocol.json")["identity"]["config"]["longbench_data"]
    longbench = {row["_id"]: row for row in load_public_samples(data_path)}
    assert len(longbench) == 125
    questions = {"math": math, "longbench": longbench}
    runs = {}
    for seed in SEEDS:
        settings = configuration(seed)
        root = Path(settings["output"])
        identity = read_json(root / "protocol.json")["identity"]
        assert identity["configuration"] == settings
        assert read_json(root / "r31_implementation.json") == implementation_identity()
        for paths in (identity["files"], identity["core"]):
            for path, expected in paths.items():
                assert sha(Path(path)) == expected, path
        runs[seed] = {}
        for dataset in DATASETS:
            result = audit_run(settings, dataset, questions[dataset], judge, deployment)
            runs[seed][dataset] = result
            print(seed, dataset, result["count"], result["score"], result["controls"], flush=True)
    root = Path(read_json(CONFIG)["output"])
    write_json(root / "summary.json", {"runs": runs, "prediction_count": sum(
        row["count"] for runs_by_dataset in runs.values() for row in runs_by_dataset.values()),
        "grade_count": sum(r["math"]["grade_count"] for r in runs.values())})
    publish(runs, root)


if __name__ == "__main__":
    main()
