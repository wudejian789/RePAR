"""Audit persistent objectives and compare 0.8B MATH scores with r29."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from statistics import mean

import jsonschema

from repar.benchmarks.math_suite import usage
from repar.io import read_json, write_json
from repar.v4 import objectives
from report_math_r28 import audit_condition, audit_grade, saved_request, tree_query_counts
from run_benchmark_recommended import sha
from run_math_r28 import questions_and_judge
from run_math_r29 import configuration as r29_configuration
from run_math_r30 import CONFIG, MODEL, configuration, implementation_identity, paired


SECTION = "## RePAR v4-r30 0.8B MATH 多目标编排"


def audit_control(folder, row, settings):
    stages = {}
    stats = Counter(tree_query_counts(row["trace"]))
    prompts = {"plan": objectives.PLAN_PROMPT, "action": objectives.ACTION_PROMPT,
               "reflection": objectives.REFLECTION_PROMPT}
    task = row["question"]
    anchor = "\n".join([task["question_stem"], *(c["text"] for c in task["choices"]),
                        *(part["question"] for part in task["subquestions"])])
    for call in row["calls"][:-1]:
        _, stage, number, *repair = call["tag"].split("_")
        record = read_json(folder / "cache/solver" / f"{call['key']}.json")
        request = record["request"]
        assert record["response"] == call["response"]
        assert request["messages"][0]["content"] == prompts[stage]
        if stage == "action":
            assert request["max_tokens"] == settings["action_max_tokens"]
        stages[stage, int(number)] = record
        stats[f"{stage}_requests"] += 1
        stats["repairs"] += bool(repair)
    plan, covered, replan = None, [], True
    objective_indices = set()
    for step in row["trace"]:
        number = step["round"]
        assert step["planned"] == replan
        if "error" in step:
            stats["invalid_controls"] += 1
        if "plan" not in step:
            assert "error" in step and not step["executed"]
            continue

        def actual(stage):
            record = stages[stage, number]
            value = json.loads(record["response"]["choices"][0]["message"]["content"])
            schema = record["request"]["response_format"]["json_schema"]["schema"]
            jsonschema.validate(value, schema)
            state = json.loads(record["request"]["messages"][1]["content"])
            return value, state, schema

        if replan:
            stats["plans_with_goal_switch"] += len(objective_indices) > 1
            objective_indices.clear()
            value, state, schema = actual("plan")
            assert schema == objectives.PLAN_SCHEMA and state["task"] == anchor
            assert state["previous_objectives"] == (plan["objectives"] if plan else [])
            assert state["previously_covered"] == covered
            plan, covered, replan = value, [], False
            stats["plans"] += 1
            stats["multi_objective_plans"] += len(plan["objectives"]) > 1
        else:
            assert ("plan", number) not in stages
            stats["plan_reuses"] += 1
        assert step["plan"] == plan and step["covered_objectives_before"] == covered
        if "action_proposal" not in step:
            assert "error" in step and not step["executed"]
            continue
        proposal, state, _ = actual("action")
        assert proposal == step["action_proposal"] and state["task"] == anchor
        assert state["objectives"] == plan["objectives"] and state["covered_objectives"] == covered
        if "blocked_before_execution" in step:
            stats["blocked_calls"] += 1
            stats["blocked_duplicates"] += step["blocked_before_execution"]["reason"] == "Duplicate completed action"
            assert not step["executed"] and ("reflection", number) not in stages
            continue
        decision = step["decision"]
        assert decision["tool"] == proposal["tool"] and decision["objective_index"] == proposal["objective_index"]
        parameters = dict(proposal["parameters"])
        if "query" in parameters:
            parameters["query"] = parameters["query"].strip() or anchor
        assert all(decision["parameters"][key] == value for key, value in parameters.items())
        if decision["tool"] == "tree_search" and parameters["query"] != anchor:
            assert decision["parameters"]["anchor_query"] == anchor
        if decision["tool"] == "answer":
            assert not step["executed"] and ("reflection", number) not in stages
            continue
        execution, = step["executed"]
        objective_indices.add(decision["objective_index"])
        assert execution["name"] == decision["tool"] and execution["parameters"] == decision["parameters"]
        stats[f"tool_{execution['name']}"] += 1
        if "reflection" not in step:
            assert "error" in step
            continue
        reflection, state, _ = actual("reflection")
        assert step["reflection"] == reflection and state["plan"] == plan
        assert state["action"]["parameters"] == decision["parameters"]
        assert state["observation"] == execution["visible_observation"]
        covered = sorted(set(reflection["covered_objectives"]))
        replan = reflection["next"] == "replan"
        stats["replans_requested"] += replan
    stats["plans_with_goal_switch"] += len(objective_indices) > 1
    stats["termination_" + row["termination"]] += 1
    return stats


def audit_seed(seed, questions, judge, deployment, partial):
    settings = configuration(seed)
    root = Path(settings["output"])
    manifest_path = root / "r30_implementation.json"
    if partial and not manifest_path.exists():
        return None
    folder = root / MODEL / "math/repar"
    previous_folder = Path(r29_configuration(seed)["output"]) / MODEL / "math/repar"
    identity = read_json(root / "protocol.json")["identity"]
    assert identity["configuration"] == settings
    assert read_json(manifest_path) == implementation_identity()
    for paths in (identity["files"], identity["core"]):
        for path, expected in paths.items():
            assert sha(Path(path)) == expected, path

    grades = {p.stem: read_json(p) for p in (folder / "grades").glob("*.json")}
    rows = {p.stem: read_json(p) for p in (folder / "cases").glob("*.json")}
    assert grades.keys() <= rows.keys() <= questions.keys()
    complete = grades.keys() == questions.keys()
    assert partial or complete, (seed, len(rows), len(grades))
    if grades:
        manifest = read_json(root / "08_judge_protocol.json")
        assert manifest["deployment"] == deployment
        assert all(manifest["config"][key] == value for key, value in judge.items() if key != "deployment_id")
        for path, expected in manifest["files"].items():
            assert sha(Path(path)) == expected, path

    stats, per_question, judge_calls = Counter(), {}, []
    for qid, row in rows.items():
        old = read_json(previous_folder / "cases" / f"{qid}.json")
        assert row["status"] == "ok" and row["agent_version"] == settings["agent_version"]
        assert row["query_tokenizer"] == "chinese"
        assert all(row[key] == old[key] for key in ("task", "context_sha256"))
        assert row["index_reuse"]["sha256"] == old["index_reuse"]["sha256"]
        answer_request = saved_request(folder, row)
        assert answer_request == dict(saved_request(previous_folder, old), messages=answer_request["messages"])
        controls = audit_control(folder, row, settings)
        stats.update(controls)
        pair = {"controls": dict(controls), "r30_finish": row["calls"][-1]["response"]["choices"][0]["finish_reason"],
                "r29_finish": old["calls"][-1]["response"]["choices"][0]["finish_reason"]}
        if qid in grades:
            grade = grades[qid]
            audit_grade(folder / "cache/judge", grade, questions[qid], row["response"], judge)
            judge_calls.extend(grade["judge_calls"])
            pair["r30_score"] = grade["score"]
            pair["r29_score"] = read_json(previous_folder / "grades" / f"{qid}.json")["score"]
        per_question[qid] = pair
    result = audit_condition(settings, root, MODEL, "math", "repar", questions, judge, False)
    if complete:
        result["score"] = sum(pair["r30_score"] for pair in per_question.values())
        result["r29_score"] = sum(pair["r29_score"] for pair in per_question.values())
    result.update(complete=complete, grade_count=len(grades), controls=dict(stats), comparisons=per_question,
                  full=sum(grade["fully_correct"] for grade in grades.values()), judge_usage=usage(judge_calls))
    return result


def publish(runs, root):
    comparisons = [pair for run in runs.values() for pair in run["comparisons"].values()]
    versions = ("r29", "r30")
    lines = [SECTION, "",
             "两版按种子 42、43、44 比较 0.8B 作答。共享中文索引、温度 0.1、4K 作答上限与 27B 温度 0 匿名阅卷。"
             "r30 的 Action 负责生成参数，控制输出上限为 256 tokens。", "",
             "| 版本 | 种子 42 | 种子 43 | 种子 44 | 均分 /150 | 4K 触顶 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for version in versions:
        scores = [sum(pair[f"{version}_score"] for pair in row["comparisons"].values()) for row in runs.values()]
        cap = sum(pair[f"{version}_finish"] == "length" for pair in comparisons)
        lines.append(f"| {version} | " + " | ".join(f"{score:.3f}" for score in scores)
                     + f" | {mean(scores):.3f} | {cap}/{len(comparisons)} |")
    seconds = mean(row["case_mean"] for row in runs.values())
    inputs = mean(row["costs"]["input"]["sum"] for row in runs.values())
    outputs = mean(row["costs"]["output"]["sum"] for row in runs.values())
    calls = sum(row["calls"] for row in runs.values())
    lines.extend(["", f"r30 作答平均 {seconds:.2f} 秒/题，每轮平均输入 {inputs:,.0f} tokens、输出 {outputs:,.0f} tokens。"
                      f"57 题作答及控制共 {calls} 次 LLM 请求，阅卷另计。",
                  "", f"[逐题评分、耗时与控制轨迹审计]({(root / 'summary.json').as_posix()})"])
    report = Path("all_results_new.md")
    body = report.read_text(encoding="utf-8").split(SECTION, 1)[0].rstrip()
    temp = report.with_suffix(".md.tmp")
    temp.write_text(body + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(report)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--partial", action="store_true")
    args = cli.parse_args()
    questions, judge, deployment = questions_and_judge()
    root = Path(read_json(CONFIG)["output"])
    runs = {}
    for seed in paired.SEEDS:
        row = audit_seed(seed, questions, judge, deployment, args.partial)
        if row is not None:
            runs[seed] = row
            print(seed, row["count"], row["grade_count"], row["score"], row["answer_finish"], row["controls"])
    complete = len(runs) == len(paired.SEEDS) and all(row["complete"] for row in runs.values())
    name = "summary.json" if complete else "progress.json"
    write_json(root / name, {"updated_at": datetime.now().astimezone().isoformat(),
                            "grade_count": sum(row["grade_count"] for row in runs.values()), "runs": runs})
    if complete:
        publish(runs, root)


if __name__ == "__main__":
    main()
