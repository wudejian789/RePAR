"""Audit the paired three-seed r32 evaluation and publish the totals."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import jsonschema

from repar.benchmarks.math_suite import usage
from repar.io import load_public_samples, read_json, write_json
from repar.v4 import objective_state, objectives
from repar.v4.knowledge import request_identity
from report_benchmark_recommended import audit_condition
from report_math_r28 import audit_grade, saved_request
from run_benchmark_r32 import CONFIG, configuration, implementation_identity, suite
from run_math_r28 import questions_and_judge


PRIOR = Path("artifacts/benchmark_r31_08_20261124")
PROMPTS = {"plan": objective_state.INQUIRY_PLAN_PROMPT, "action": objective_state.SEARCH_ACTION_PROMPT,
           "reflection": objective_state.QUOTED_REFLECTION_PROMPT}
SECTION = "## RePAR v4-r32 0.8B 引文核验与检索控制"
DATASET_NAMES = {"math": "MATH", "longbench": "LongBench v2 Multi-Document"}


def audit_trace(folder, row, settings, config):
    stats = Counter()
    requests = {}
    for call in row["calls"][:-1]:
        _, stage, number, *repair = call["tag"].split("_")
        record = read_json(folder / "cache/solver" / f"{call['key']}.json")
        assert call["response"] == record["response"]
        assert record["request"]["messages"][0]["content"] == PROMPTS[stage]
        if stage in {"plan", "action"}:
            assert record["request"]["max_tokens"] == settings[f"{stage}_max_tokens"]
        requests.setdefault((stage, int(number)), []).append(record)
        stats[f"{stage}_requests"] += 1
        stats["repairs"] += bool(repair)
        stats[f"{stage}_length"] += record["response"]["choices"][0]["finish_reason"] == "length"

    def read_control(stage, number, step):
        attempts = requests.pop((stage, number))
        record = attempts[-1]
        request = record["request"]
        schema = request["response_format"]["json_schema"]["schema"]
        payload = json.loads(request["messages"][1]["content"])
        for attempt, item in enumerate(attempts, 1):
            raw = item["response"]["choices"][0]["message"]["content"]
            try:
                value = json.loads(raw)
                json.dumps(value, ensure_ascii=False).encode("utf-8")
                jsonschema.validate(value, schema)
            except (ValueError, TypeError, jsonschema.ValidationError):
                if attempt == len(attempts):
                    assert step["error"] == f"Invalid structured output: v4_{stage}_{number}"
                    return None, payload, schema
            else:
                assert attempt == len(attempts)
                return value, payload, schema

    goals = objective_state.ObjectiveState(quote_evidence=True)
    needs_plan, completed = True, set()
    for round_number, step in enumerate(row["trace"], 1):
        assert step["protocol"] == "quoted_objective_state_v4"
        assert step["round"] == round_number and step["planned"] == needs_plan
        if needs_plan:
            plan, payload, schema = read_control("plan", round_number, step)
            if plan is None:
                break
            assert plan == step["plan_proposal"]
            assert payload["objective_state"] == goals.snapshot()
            assert goals.plan_request(payload)[1] == schema
            stats["replans"] += round_number > 1
            try:
                goals.apply_plan(plan)
            except ValueError as error:
                assert str(error) == step["error"]
                break
            needs_plan = False
        assert step["objective_state_before"] == goals.snapshot()
        stats["action_with_all_covered"] += not goals.pending()
        action, payload, schema = read_control("action", round_number, step)
        if action is None:
            break
        assert payload["objective_state"] == goals.snapshot() and payload["pending_objectives"] == goals.pending()
        assert step["action_proposal"] == action
        for branch in schema["oneOf"]:
            fields = branch["properties"]
            if fields["tool"]["const"] != "answer":
                assert fields["objective_id"]["enum"] == goals.active
            assert "top_k" not in fields["parameters"]["required"]
        if "blocked_before_execution" in step:
            blocked = step["blocked_before_execution"]
            assert not step["executed"] and blocked["proposal"] == action
            stats["blocked_actions"] += 1
            stats["duplicate_actions"] += blocked["reason"] == "Duplicate completed action"
            continue

        decision = step["decision"]
        assert decision["tool"] == action["tool"] and decision.get("objective_id") == action.get("objective_id")
        if action["tool"] == "answer":
            assert not step["executed"]
            stats["action_answer"] += 1
            continue
        execution, = step["executed"]
        assert decision["gap"] == goals.records[action["objective_id"]]["gap"]
        assert execution["name"] == decision["tool"] and execution["parameters"] == decision["parameters"]
        memory = SimpleNamespace(records=payload["source_ranges"])
        bound = objectives.bind_action(action, payload["task"], memory, config)
        assert bound["parameters"] == decision["parameters"]
        if "top_k" in decision["parameters"]:
            stats["defaulted_top_k"] += "top_k" not in action["parameters"]
            stats["effective_top_k_one"] += decision["parameters"]["top_k"] == 1
            stats["searches_with_top_k"] += 1
        identity = request_identity(execution["name"], execution["parameters"])
        assert identity not in completed
        completed.add(identity)
        stats["tool_executions"] += 1
        stats[f"tool_{execution['name']}"] += 1
        stats["invalid_tool_executions"] += execution["observation"]["status"] == "invalid"

        reflection, payload, schema = read_control("reflection", round_number, step)
        if reflection is None:
            stats["failed_reflection_requests"] += 1
            break
        feedback = step["reflection_context"]
        assert reflection == step["reflection"] and payload["objective_state"] == goals.snapshot()
        assert payload["observation"] == execution["visible_observation"]
        assert all(payload[key] == feedback[key] for key in ("existing_sources", "source_quotations"))
        assert goals.reflection_request(payload)[1] == schema
        rejected = goals.apply_reflection(reflection, payload)
        assert rejected == step.get("rejected_objective_updates", [])
        assert step["objective_state_after"] == goals.snapshot()
        stats["rejected_updates"] += len(rejected)
        stats["proposed_updates"] += len(reflection["updates"])
        stats["accepted_covered_updates"] += sum(
            update["status"] == "covered" for update in reflection["updates"] if update not in rejected)
        needs_plan = reflection["next"] == "replan"
        deferred = bool(rejected) and reflection["next"] == "answer"
        assert step.get("control_transition") == ("answer_deferred_after_rejected_updates" if deferred else None)
        stats["answers_deferred"] += deferred
        stats["continue_all_covered"] += reflection["next"] == "continue" and not goals.pending()
    assert not requests
    assert row["state"]["objective_state"] == goals.snapshot()
    assert set(row["state"]["completed_actions"]) == completed
    stats[f"finish_{row['termination']}"] += 1
    stats["single_tool_cases"] += len(completed) == 1
    return stats


def audit_run(settings, dataset, questions, judge, deployment, prior_run, config):
    model = suite.MODEL
    root = Path(settings["output"])
    folder = root / model / dataset / "repar"
    prior = PRIOR / f"seed_{settings['answer_seed']}" / model / dataset / "repar"
    is_math = dataset == "math"
    rows = {p.stem: read_json(p) for p in (folder / "cases").glob("*.json")}
    assert rows.keys() == questions.keys() == prior_run["comparisons"].keys()
    assert all(row["id"] == qid for qid, row in rows.items())
    grades = {}
    if is_math:
        grades = {p.stem: read_json(p) for p in (folder / "grades").glob("*.json")}
        assert grades.keys() == questions.keys()
        manifest = read_json(root / "08_judge_protocol.json")
        assert manifest["deployment"] == deployment
        assert all(manifest["config"][key] == value for key, value in judge.items() if key != "deployment_id")
        for filename, expected in manifest["files"].items():
            assert suite.runner.sha(Path(filename)) == expected, filename

    result = audit_condition(settings, root, model, dataset, "repar", questions, judge, not is_math)
    controls, comparisons, judge_calls = Counter(), {}, []
    for qid, row in rows.items():
        prior_row = read_json(prior / "cases" / f"{qid}.json")
        assert row["status"] == "ok" and row["agent_version"] == settings["agent_version"]
        assert all(row[key] == prior_row[key] for key in ("task", "context_sha256", "base_url", "query_tokenizer"))
        assert row["index_reuse"]["path"] == prior_row["index_reuse"]["path"]
        if is_math:
            assert row["index_reuse"]["sha256"] == prior_row["index_reuse"]["sha256"]
        body = saved_request(folder, row)
        assert body == {**saved_request(prior, prior_row), "messages": body["messages"]}
        stats = audit_trace(folder, row, settings, config)
        controls.update(stats)
        if is_math:
            grade = grades[qid]
            assert grade["id"] == qid
            audit_grade(folder / "cache/judge", grade, questions[qid], row["response"], judge)
            judge_calls.extend(grade["judge_calls"])
        prior_pair = prior_run["comparisons"][qid]
        comparisons[qid] = {"score": grades[qid]["score"] if is_math else row["correct"],
                            "prior_score": prior_pair["score"], "controls": dict(stats),
                            "finish": row["calls"][-1]["response"]["choices"][0]["finish_reason"],
                            "prior_finish": prior_pair["finish"], "final_sources": len(row["evidence_ids"]),
                            "prior_final_sources": len(prior_row["evidence_ids"])}
    result.update(complete=True, grade_count=len(grades), score=sum(p["score"] for p in comparisons.values()),
                  prior_score=prior_run["score"], prior_answer_finish=prior_run["answer_finish"],
                  prior_controls=prior_run["controls"], controls=dict(controls),
                  judge_usage=usage(judge_calls), comparisons=comparisons)
    return result


def publish(runs, root):
    lines = [SECTION, "",
             "Qwen3.5-0.8B 作答，使用种子 42、43、44 与 r31 配对。MATH 复用中文索引，温度 0.1，"
             "4,096-token 自由文本作答，27B 温度 0 匿名阅卷。LongBench 复用历史索引，"
             "温度 0.1，128-token 上限，四选一 GBNF。", "",
             "| 评测 | r31 三轮 | r32 三轮 | r31 均值 | r32 均值 |",
             "| --- | --- | --- | ---: | ---: |"]
    for dataset, label in DATASET_NAMES.items():
        rows = [record[dataset] for record in runs.values()]
        previous, current = ([row[key] for row in rows] for key in ("prior_score", "score"))
        denominator = 150 if dataset == "math" else 125
        fmt = ".3f" if dataset == "math" else ".0f"
        old_scores, new_scores = ("、".join(format(value, fmt) for value in scores)
                                  for scores in (previous, current))
        lines.append(f"| {label} | {old_scores} | {new_scores} | "
                     f"{mean(previous):.3f}/{denominator} | {mean(current):.3f}/{denominator} |")

    lines.extend(["", "| 评测 | r32 秒/题 | 输入 token/轮 | 输出 token/轮 | 作答与控制请求 |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for dataset, label in DATASET_NAMES.items():
        rows = [record[dataset] for record in runs.values()]
        lines.append(f"| {label} | {mean(row['case_mean'] for row in rows):.2f} | "
                     f"{mean(row['costs']['input']['sum'] for row in rows):,.0f} | "
                     f"{mean(row['costs']['output']['sum'] for row in rows):,.0f} | {sum(row['calls'] for row in rows):,} |")
    math = [record["math"] for record in runs.values()]
    caps = [sum(row[key].get("length", 0) for row in math) for key in ("prior_answer_finish", "answer_finish")]
    lines.extend(["", f"MATH 的 4K 触顶由 r31 的 {caps[0]}/57 变为 r32 的 {caps[1]}/57。"
                  "时间和 token 含在线控制及作答，匿名阅卷另计。", "",
                  f"[逐题分差、耗时与控制状态审计]({(root / 'summary.json').as_posix()})"])
    report = Path("all_results_new.md")
    existing = report.read_text(encoding="utf-8")
    before, heading, after = existing.partition(SECTION)
    following = "\n## " + after.split("\n## ", 1)[1] if heading and "\n## " in after else ""
    temp = report.with_name("all_results_new.r32.tmp")
    temp.write_text(before.rstrip() + "\n\n" + "\n".join(lines) + "\n" + following, encoding="utf-8")
    temp.replace(report)


def main():
    math, judge, deployment = questions_and_judge()
    original = read_json(suite.runner.OLD[suite.MODEL] / "protocol.json")["identity"]
    longbench = {row["_id"]: row for row in load_public_samples(original["config"]["longbench_data"])}
    assert len(math) == 19 and len(longbench) == 125
    questions = {"math": math, "longbench": longbench}
    prior = read_json(PRIOR / "summary.json")["runs"]
    runs = {}
    for seed in suite.SEEDS:
        settings = configuration(seed)
        folder = Path(settings["output"])
        identity = read_json(folder / "protocol.json")["identity"]
        assert identity["configuration"] == settings
        assert read_json(folder / "r32_implementation.json") == implementation_identity()
        for paths in (identity["files"], identity["core"]):
            for filename, expected in paths.items():
                assert suite.runner.sha(Path(filename)) == expected, filename
        runs[seed] = {}
        config = {**original["solver"], "agent_version": settings["agent_version"]}
        for dataset in suite.DATASETS:
            result = audit_run(settings, dataset, questions[dataset], judge, deployment, prior[str(seed)][dataset],
                               config)
            runs[seed][dataset] = result
            print(seed, dataset, result["count"], result["score"], result["controls"], flush=True)
    root = Path(read_json(CONFIG)["output"])
    predictions = sum(run["count"] for record in runs.values() for run in record.values())
    grades = sum(record["math"]["grade_count"] for record in runs.values())
    assert predictions == 432 and grades == 57
    write_json(root / "summary.json", {"runs": runs, "prediction_count": predictions, "grade_count": grades})
    publish(runs, root)


if __name__ == "__main__":
    main()
