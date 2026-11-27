"""Audit the new 27B MATH judgments and publish the revised benchmark table."""
from argparse import ArgumentParser
from pathlib import Path

from repar.benchmarks.math_exam import aggregate_grades, score_choice
from repar.benchmarks.math_judge import judge_messages, judge_schema, validate_grade
from repar.benchmarks.math_suite import usage
from repar.io import digest, read_json, write_json

from recover_math_2025_judge_quotes import verify_recovery
from regrade_math_2025 import CHOICE_BUDGET, CHOICE_TYPES, ROOT, choice_input, choice_score, protocol, read_sources
from repeat_benchmark_recommended import REPEAT_ROOT, SEEDS, configuration
from report_benchmark_recommended import REPORT, historical_cells, render, render_repeats
from resume_math_2025_regrade import choice_observation


MATH_JUDGE_NOTE = (
    "MATH 使用 27B 温度 0 匿名阅卷。单选和填空整题全分或零分。"
    "多选全对得 6 分，正确非空子集按正确项比例折算，有错选得零分。"
    "证明及解答题按小问给部分分。选择题新评，填空和解答题沿用同一 27B 阅卷的已核验记录。"
)


def score_judged_choice(question, prediction):
    # Apply the exam rubric to the judge's validated selection, not the raw candidate text.
    return score_choice(question, f"最终答案：{prediction}" if prediction else "")


def audit_requests(cache, calls, judge, messages, schema, budget, replicas):
    responses_by_seed = {}
    for call in calls:
        record = read_json(cache / f"{call['key']}.json")
        body = record["request"]
        assert record["response"] == call["response"]
        assert body["model"] == judge["model"] and body["temperature"] == judge["temperature"] == 0
        assert body["max_tokens"] == budget and body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["messages"][:len(messages)] == messages
        assert body["response_format"]["json_schema"]["schema"] == schema
        responses_by_seed[body["seed"]] = record["response"]["choices"][0]["message"]["content"]
    assert list(responses_by_seed) == judge["replicate_seeds"][:replicas]
    return list(responses_by_seed.values())


def audit_condition(folder, source, questions, judge, partial):
    original, predictions, original_grades = source
    rows = {}
    for path in sorted((folder / "grades").glob("*.json")):
        qid = path.stem
        question = questions[qid]
        grade = read_json(path)
        response = predictions[qid]["response"]
        assert grade["id"] == qid and grade["response_sha256"] == digest(response)
        if question["question_type"] in CHOICE_TYPES:
            messages, schema = choice_input(question, response)
            raws = audit_requests(folder / "cache", grade["judge_calls"], judge, messages, schema, CHOICE_BUDGET,
                                  len(grade["observations"]))
            results = [choice_observation(question, response, raw) for raw in raws]
            assert results == grade["observations"]
            observed = choice_score(question, results)
            assert all(grade[key] == observed[key] for key in ("prediction", "score", "fully_correct", "review_flag"))
            grade.update(score_judged_choice(question, observed["prediction"]))
        else:
            assert grade == original_grades[qid]
            audit_requests(original / "cache/judge", grade["judge_calls"], judge, judge_messages(question, response),
                           judge_schema(question), judge["max_tokens"], len(grade["grades"]))
            expected = aggregate_grades(question, [validate_grade(question, r) for r in grade["grades"]])
            if question["question_type"] == "fill_in_blank":
                assert grade["score"] in {0, question["score"]}
            assert all(grade[key] == expected[key] for key in ("score", "fully_correct", "review_flag"))
        rows[qid] = grade
    if not partial:
        assert rows.keys() == questions.keys(), (folder, len(rows))
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    questions, sources, files = read_sources()
    judge = protocol(files, online=False)
    verify_recovery()
    baseline_path = REPEAT_ROOT / "summary.json"
    baseline = read_json(baseline_path)
    summaries = {seed: baseline["runs"][str(seed)] for seed in SEEDS}
    previous, regraded, new_calls, reused_calls = {}, {}, [], []
    for (seed, model, method), source in sources.items():
        folder = ROOT / str(seed) / model / method
        group = f"{model}/math"
        original = summaries[seed][group][method]
        assert original["complete"] and original["count"] == 19
        assert set(original["per_question"]) == questions.keys()
        assert original["score"] == sum(g["score"] for g in source[2].values())
        rows = audit_condition(folder, source, questions, judge, args.allow_partial)
        key = f"{seed}/{model}/{method}"
        previous[key] = original["score"]
        regraded[key] = {"count": len(rows), "grades": rows}
        for qid, grade in rows.items():
            calls = new_calls if questions[qid]["question_type"] in CHOICE_TYPES else reused_calls
            calls.extend(grade["judge_calls"])
        complete = len(rows) == 19
        original.update(complete=complete, grade_count=len(rows),
                        score=sum(row["score"] for row in rows.values()) if complete else None,
                        full=sum(row["fully_correct"] for row in rows.values()) if complete else None,
                        judge_usage=usage([call for row in rows.values() for call in row["judge_calls"]]),
                        judge_disagreements=sum(len(row.get("observations", row.get("grades", []))) == 3 for row in rows.values()))
    build = historical_cells()
    details = {seed: render(configuration(seed), summaries[seed], build, math_judge_note=MATH_JUDGE_NOTE) for seed in SEEDS}
    destination = ROOT / "summary.json"
    lines, means = render_repeats(summaries, details, math_judge_note=MATH_JUDGE_NOTE, summary_path=destination.as_posix())
    count = sum(row["count"] for row in regraded.values())
    write_json(destination, {"seeds": SEEDS, "means": means, "runs": summaries, "previous_scores": previous,
                             "grading_rubric": {"single_choice": "exact match", "fill_in_blank": "complete mathematical equivalence",
                                                "multiple_choice": "correct nonempty subset proportional to full credit; incorrect selection gets zero",
                                                "constructed_response": "mean of median per-subquestion credits"},
                             "math_grades": regraded, "new_choice_judge_usage": usage(new_calls),
                             "reused_open_ended_judge_usage": usage(reused_calls), "graded_count": count})
    if count == 399:
        assert len(new_calls) >= 462 and len(reused_calls) >= 336
        temporary = REPORT.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(REPORT)
    print(f"Audited {count}/399 MATH judgments", flush=True)


if __name__ == "__main__":
    main()
