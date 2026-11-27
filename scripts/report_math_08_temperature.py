"""Audit and publish the paired 0.8B MATH temperature results."""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from statistics import mean

from repar.benchmarks.math_exam import aggregate_grades
from repar.benchmarks.math_judge import judge_messages, judge_schema, validate_grade
from repar.benchmarks.math_suite import usage
from repar.io import digest, read_json, write_json

from grade_math_08_temperature import questions_and_judge
from regrade_math_2025 import CHOICE_BUDGET, CHOICE_TYPES, ROOT as REFERENCE, choice_input, choice_score
from repeat_benchmark_recommended import configuration as reference_configuration
from report_benchmark_recommended import METHODS, REPORT, audit_condition, historical_cells, render, render_repeats
from report_math_2025_regrade import audit_requests, score_judged_choice
from resume_math_2025_regrade import choice_observation
from run_benchmark_recommended import sha
from run_math_08_temperature import CONFIG_PATH, SEEDS, configuration, generation_identity


def saved_request(folder, record):
    call = record["calls"][-1]
    value = read_json(folder / "cache/solver" / f"{call['key']}.json")
    assert value["response"] == call["response"]
    assert (value["response"]["choices"][0]["message"]["content"] or "") == record["response"]
    return value["request"]


def audit_marks(folder, reference, questions, judge, method, settings):
    rows = {}
    for path in sorted((folder / "grades").glob("*.json")):
        question = questions[path.stem]
        record = read_json(folder / "cases" / path.name)
        previous = read_json(reference / "cases" / path.name)
        old_request = saved_request(reference, previous)
        request = saved_request(folder, record)
        assert record["context_sha256"] == previous["context_sha256"]
        expected_request = dict(old_request, temperature=settings["sampling"]["08"]["temperature"])
        if method == "repar":
            expected_request["messages"] = request["messages"]
            assert record["index_reuse"]["sha256"] == previous["index_reuse"]["sha256"]
            assert record["query_tokenizer"] == "chinese" and previous["index_reuse"]["stats"]["extractor"]["model"] == "zh_core_web_sm"
        assert request == expected_request, (method, path.stem)
        grade = read_json(path)
        answer = record["response"]
        assert grade["id"] == path.stem and grade["response_sha256"] == digest(answer)
        cache = folder / "cache/judge"
        if question["question_type"] in CHOICE_TYPES:
            messages, schema = choice_input(question, answer)
            raws = audit_requests(cache, grade["judge_calls"], judge, messages, schema, CHOICE_BUDGET, len(grade["observations"]))
            actual = [choice_observation(question, answer, raw) for raw in raws]
            assert actual == grade["observations"]
            vote = choice_score(question, actual)
            expected = {**vote, **score_judged_choice(question, vote["prediction"])}
            assert grade["prediction"] == expected["prediction"]
        else:
            audit_requests(cache, grade["judge_calls"], judge, judge_messages(question, answer),
                           judge_schema(question), judge["max_tokens"], len(grade["grades"]))
            expected = aggregate_grades(question, [validate_grade(question, value) for value in grade["grades"]])
        assert all(grade[key] == expected[key] for key in ("score", "fully_correct", "review_flag"))
        rows[path.stem] = grade
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    questions, judge, deployment = questions_and_judge()
    experiment = read_json(CONFIG_PATH)
    root = Path(experiment["output"])
    baseline = read_json(REFERENCE / "summary.json")
    assert baseline["graded_count"] == 399
    judge_manifest = read_json(root / "judge_protocol.json")
    assert judge_manifest["judge"] == judge and judge_manifest["deployment"] == deployment
    for path, expected in judge_manifest["files"].items():
        assert sha(Path(path)) == expected
    summaries = {seed: deepcopy(baseline["runs"][str(seed)]) for seed in SEEDS}
    grades, details = {}, {}
    generation_note = (f"0.8B MATH 的作答和 r27 在线控制温度为 {experiment['temperature']}，其余采样沿用模型卡推荐设置。"
                       "27B MATH 沿用模型卡作答采样。")
    judge_note = ("MATH 各题型由 27B 温度 0 匿名阅卷。单选和填空按全分或零分计，多选对正确非空子集给比例部分分，"
                  "有错选得零分。证明及解答题按小问给分。")
    build = historical_cells()
    for seed in SEEDS:
        settings = configuration(seed)
        seed_root = Path(settings["output"])
        source_settings = reference_configuration(seed)
        source_root = Path(source_settings["output"])
        proto = seed_root / "protocol.json"
        if args.allow_partial and not proto.exists():
            continue
        assert read_json(seed_root / "temperature_implementation.json") == generation_identity(settings)
        frozen = read_json(proto)
        assert frozen["identity"]["configuration"] == settings
        for path, expected in frozen["identity"]["files"].items():
            assert sha(Path(path)) == expected, path
        new_conditions = {}
        for method in settings["conditions"]["08/math"]:
            grade_path = Path("08/math") / method
            cases = audit_marks(seed_root / grade_path, source_root / grade_path, questions, judge, method, settings)
            assert cases.keys() <= questions.keys()
            if not args.allow_partial:
                assert cases.keys() == questions.keys(), (seed, method, len(cases))
            trial = audit_condition(settings, seed_root, "08", "math", method, questions, judge, False,
                                    first_settings=configuration(SEEDS[0]))
            complete = len(cases) == len(questions)
            if complete:
                assert trial["count"] == 19
            grade_calls = [call for record in cases.values() for call in record["judge_calls"]]
            trial.update(complete=complete, grade_count=len(cases),
                         score=sum(cases[qid]["score"] for qid in questions) if complete else None,
                         full=sum(cases[qid]["fully_correct"] for qid in questions) if complete else None,
                         judge_usage=usage(grade_calls),
                         judge_disagreements=sum(len(row.get("observations", row.get("grades", []))) == 3 for row in cases.values()))
            grades[f"{seed}/{method}"] = cases
            new_conditions[method] = trial
        summaries[seed]["08/math"] = new_conditions
        details[seed] = render(source_settings, summaries[seed], build, math_judge_note=judge_note, math_generation_note=generation_note)
    count = sum(map(len, grades.values()))
    destination = root / "summary.json"
    if count == 228:
        temperature_1_means = {method: baseline["means"]["08/math"][method]["score"]
                               for method in summaries[42]["08/math"]}
        lines, means = render_repeats(summaries, details, math_judge_note=generation_note + judge_note, summary_path=destination.as_posix(),
                                      temperature_1_means=temperature_1_means)
        lines[0] = "# 两种基模评测 · MATH 温度配对"
        lines[4:4] = ["0.8B MATH 并列温度 1.0 的三轮均值和温度 0.1 的逐种子新结果。"
                      "本次重跑共核验 228 份作答。其余条件采用此前已核验的三轮结果。", "",
                      "[温度覆盖配置](configs/math_08_temperature_01_20261124.json) · "
                      "[配对评测记录](docs/2026-11-24_1225_0.8B数学温度回退配对.md)", ""]
        temperature_scores = {}
        for method, result in means["08/math"].items():
            paired = [summaries[seed]["08/math"][method]["score"] - baseline["runs"][str(seed)]["08/math"][method]["score"]
                      for seed in SEEDS]
            temperature_scores[method] = {"temperature_1_mean": baseline["means"]["08/math"][method]["score"],
                                           "temperature_01_mean": result["score"], "paired_deltas": paired, "mean_delta": mean(paired)}
        write_json(destination, {"graded_count": count, "means": means, "runs": summaries, "math_grades": grades,
                                 "paired_temperature_comparison": temperature_scores,
                                 "reference_summary": (REFERENCE / "summary.json").as_posix()})
        temporary = REPORT.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(REPORT)
    else:
        write_json(destination, {"graded_count": count, "runs": summaries, "math_grades": grades})
    progress = ["# 0.8B MATH 温度 0.1 重跑进度", "",
                f"更新时间为 {datetime.now().astimezone():%Y-%m-%d %H:%M}（Asia/Shanghai）。已审计 {count}/228 份作答。", "",
                "| 种子 | 方法 | 已作答 | 已核分 | 该轮成绩 /150 |", "| --- | --- | ---: | ---: | ---: |"]
    for seed in SEEDS:
        for method in configuration(seed)["conditions"]["08/math"]:
            report = summaries[seed]["08/math"][method] if seed in details else {}
            score = f"{report['score']:.3f}" if report.get("complete") else "待完成"
            progress.append(f"| {seed} | {METHODS[method]} | {report.get('count', 0)}/19 | {report.get('grade_count', 0)}/19 | {score} |")
    (root / "2026-11-24_1225_温度重跑进度.md").write_text("\n".join(progress) + "\n", encoding="utf-8")
    print(f"Audited {count}/228 temperature-comparison scores", flush=True)


if __name__ == "__main__":
    main()
