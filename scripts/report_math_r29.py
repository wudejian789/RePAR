"""Audit 0.8B r29 MATH results against the same seeds in r27 and r28."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from statistics import mean

from repar.benchmarks.math_suite import usage
from repar.io import read_json, write_json
from repar.v4.workflow import PLAN_SCHEMA, SOURCE_LANGUAGE_PLAN_PROMPT

from report_math_r28 import REFERENCE_08, audit_condition, audit_grade, saved_request, tree_query_counts
from run_benchmark_recommended import sha
from run_math_r28 import questions_and_judge
from run_math_r29 import CONFIG, MODEL, configuration, implementation_identity, paired


R28 = Path("artifacts/math_r28_20261124")
SECTION = "## RePAR v4-r29 0.8B MATH 同语言查询改写"


def audit_queries(folder, row):
    task = row["question"]
    anchor = "\n".join([task["question_stem"], *(item["text"] for item in task["choices"]),
                        *(item["question"] for item in task["subquestions"])])
    plans = {}
    for call in row["calls"]:
        if call["tag"].startswith("v4_plan_"):
            saved = read_json(folder / "cache/solver" / f"{call['key']}.json")
            body = saved["request"]
            assert body["messages"][0]["content"] == SOURCE_LANGUAGE_PLAN_PROMPT
            assert body["response_format"]["json_schema"]["schema"] == PLAN_SCHEMA
            assert json.loads(body["messages"][1]["content"])["task"] == anchor
            assert saved["response"] == call["response"]
            plans[int(call["tag"].split("_")[2])] = saved["response"]["choices"][0]["message"]["content"]

    stats = Counter(tree_query_counts(row["trace"]))
    for step in row["trace"]:
        if "plan" not in step:
            assert "error" in step and not step["executed"]
            continue
        plan = json.loads(plans[step["round"]])
        focus = plan["query"].strip() or anchor
        assert step["plan"] == {**plan, "query": focus}
        for item in step["executed"]:
            if item["name"] != "tree_search":
                continue
            assert item["parameters"]["query"] == focus
            observation = item["observation"]
            if observation["status"] == "invalid":
                stats["invalid_search"] += 1
                continue
            channels = observation["query_channels"]
            expected = [("focus", focus)]
            if focus != anchor:
                assert item["parameters"]["anchor_query"] == anchor
                expected.append(("task", anchor))
            assert [(c["channel"], c["parameters"]["query"]) for c in channels] == expected
            assert [event["parameters"] for event in observation["query_execution"]] == [
                c["parameters"] for c in channels]
            if len(channels) == 2:
                stats["dual_searches"] += 1
                focus_ids, anchor_ids = (set(channel["ids"]) for channel in channels)
                anchor_only = anchor_ids - focus_ids
                returned = {record["id"] for record in observation["results"]}
                stats["anchor_recovers_empty_focus"] += not focus_ids and bool(anchor_ids)
                stats["anchor_only_results"] += len(anchor_only & returned)
                stats["anchor_only_in_final_evidence"] += len(anchor_only & returned & set(row["evidence_ids"]))
    return stats


def audit_seed(seed, questions, judge, deployment, partial):
    settings = configuration(seed)
    root = Path(settings["output"])
    folder = root / MODEL / "math/repar"
    manifest_path = root / "r29_implementation.json"
    if partial and not manifest_path.exists():
        return None
    frozen = read_json(root / "protocol.json")["identity"]
    assert frozen["configuration"] == settings
    assert read_json(manifest_path) == implementation_identity()
    for paths in (frozen["files"], frozen["core"]):
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

    refs = {"r27": REFERENCE_08 / root.name / "08/math/repar", "r28": R28 / root.name / "08/math/repar"}
    query_stats, comparisons, judge_calls = Counter(), {}, []
    for qid, row in rows.items():
        assert row["status"] == "ok" and row["agent_version"] == settings["agent_version"]
        assert row["query_tokenizer"] == "chinese"
        previous = {version: read_json(path / "cases" / f"{qid}.json") for version, path in refs.items()}
        for old in previous.values():
            assert all(row[key] == old[key] for key in ("task", "context_sha256"))
            assert row["index_reuse"]["sha256"] == old["index_reuse"]["sha256"]
        body = saved_request(folder, row)
        assert body == dict(saved_request(refs["r28"], previous["r28"]), messages=body["messages"])
        query_stats.update(audit_queries(folder, row))
        comparison = {
            "finish": {version: item["calls"][-1]["response"]["choices"][0]["finish_reason"]
                       for version, item in {**previous, "r29": row}.items()},
            "evidence_ids": {version: item["evidence_ids"] for version, item in {**previous, "r29": row}.items()},
        }
        if qid in grades:
            grade = grades[qid]
            audit_grade(folder / "cache/judge", grade, questions[qid], row["response"], judge)
            judge_calls.extend(grade["judge_calls"])
            comparison["scores"] = {version: read_json(path / "grades" / f"{qid}.json")["score"]
                                    for version, path in refs.items()}
            comparison["scores"]["r29"] = grade["score"]
        comparisons[qid] = comparison

    result = audit_condition(settings, root, MODEL, "math", "repar", questions, judge, False)
    if complete:
        result["scores"] = {version: sum(item["scores"][version] for item in comparisons.values())
                            for version in ("r27", "r28", "r29")}
    result.update(complete=complete, grade_count=len(grades), query_stats=dict(query_stats),
                  full=sum(g["fully_correct"] for g in grades.values()),
                  judge_usage=usage(judge_calls), comparisons=comparisons)
    return result


def publish(runs, root):
    versions = ("r27", "r28", "r29")
    lines = [SECTION, "",
             "0.8B 使用中文索引重测 19 题，温度 0.1，最终作答上限 4,096 tokens。"
             "同一 27B 温度 0 阅卷协议提供匿名评分。时间和 token 统计含在线控制及作答，阅卷另计。", "",
             "| 版本 | 种子 42 | 种子 43 | 种子 44 | 三轮均分 /150 | 4K 触顶 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for version in versions:
        scores = [row["scores"][version] for row in runs.values()]
        cap = sum(item["finish"][version] == "length"
                  for row in runs.values() for item in row["comparisons"].values())
        cells = " | ".join(f"{score:.3f}" for score in scores)
        lines.append(f"| {version} | {cells} | {mean(scores):.3f} | {cap}/57 |")
    seconds = mean(row["case_mean"] for row in runs.values())
    inputs = mean(row["costs"]["input"]["sum"] for row in runs.values())
    outputs = mean(row["costs"]["output"]["sum"] for row in runs.values())
    lines.extend(["", f"r29 逐题作答平均 {seconds:.2f} 秒，每轮平均输入 {inputs:,.0f} tokens、"
                      f"输出 {outputs:,.0f} tokens。", "",
                  f"[请求、查询通道及逐题评分]({(root / 'summary.json').as_posix()})"])
    report = Path("all_results_new.md")
    original = report.read_text(encoding="utf-8").split(SECTION, 1)[0].rstrip()
    temp = report.with_suffix(".md.tmp")
    temp.write_text(original + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
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
            print(seed, row["count"], row["grade_count"], row.get("scores"), row["answer_finish"], row["query_stats"])
    complete = len(runs) == len(paired.SEEDS) and all(row["complete"] for row in runs.values())
    name = "summary.json" if complete else "progress.json"
    write_json(root / name, {"updated_at": datetime.now().astimezone().isoformat(),
                            "grade_count": sum(row["grade_count"] for row in runs.values()), "runs": runs})
    if complete:
        publish(runs, root)


if __name__ == "__main__":
    main()
