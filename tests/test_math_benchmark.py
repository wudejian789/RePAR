"""Independent fixtures for label isolation, equivalence judging and coverage."""
import json
from copy import deepcopy

import pytest

from repar.benchmarks.math_exam import (aggregate_grades, choice_prediction, public_question,
                                       score_choice, stitch_context, task_text)
from repar.benchmarks.math_judge import calibration_cases, grade_once, judge_messages, validate_grade
from repar.benchmarks.math_suite import count_prompt_tokens, unique_rows


def fixture(kind="single_choice"):
    return {"id": "synthetic", "question_stem": "选出偶数。", "question_type": kind,
            "choices": [{"tag": "A", "text": "7"}, {"tag": "B", "text": "12"}],
            "score": 6, "answer_key": "B", "answer_values": [],
            "reference_solution": "SECRET_REFERENCE", "source_notes": ["SECRET_SOURCE"],
            "subquestions": [{"id": "i", "question": "求值。", "answer": "SECRET_SUBANSWER"}],
            "tables": [{"title": "样表", "columns": ["x"], "rows": [[7]], "source_original_rows": ["SECRET_TABLE"]}],
            "figures": [{"description": "四边形", "answer": "SECRET_FIGURE"}]}


def grade(credits, review=False):
    return {"units": [{"id": k, "checks": [{"claim": "claim", "verdict": "valid", "verification": "checked"}],
                       "credit": v, "reason": "Verified independently"} for k, v in credits.items()],
            "needs_review": review, "reference_check": "Equivalent or error identified"}


def test_public_allowlist_blocks_nested_answers_and_preserves_problem():
    item = fixture()
    original = deepcopy(item)
    text = task_text(item)
    assert "SECRET" not in text and '"rows": [[7]]' in text
    assert public_question(item)["subquestions"] == [{"id": "i", "question": "求值。"}]
    assert item == original


def test_choice_parser_cannot_use_intermediate_mentions_or_loose_text():
    assert choice_prediction("A错B对C错\n最终答案： B。", "single_choice") == "B"
    assert choice_prediction("最终答案：C，A", "multiple_choice") == "AC"
    assert choice_prediction("最终答案：Ｂ", "single_choice") == "B"
    for text in ("B", "答案是B", "最终答案：B或C", "最终答案：BB", "最终答案：B (because A is wrong)"):
        assert choice_prediction(text, "single_choice") is None
    assert choice_prediction("最终答案：A\n改正\n最终答案：B", "single_choice") == "B"


def test_multiple_choice_exactness_and_explicit_partial_proxy():
    item = fixture("multiple_choice")
    item["answer_key"] = "ACD"
    assert score_choice(item, "最终答案：A")['score'] == 2
    assert not score_choice(item, "最终答案：AC")["fully_correct"]
    assert score_choice(item, "最终答案：AC")["strict_score"] == 0
    assert score_choice(item, "最终答案：ABCD")["score"] == 0
    assert score_choice(item, "最终答案：DCA")["fully_correct"]


def test_context_assembly_is_lossless_ordered_and_does_not_filter_quality(tmp_path):
    source = tmp_path / "book.jsonl"
    rows = [{"chunk_id": "first", "document_id": "d", "content": " Alpha\ntext ", "quality_status": "failed"},
            {"chunk_id": "second", "document_id": "d", "content": " beta ", "quality_status": "passed"}]
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    context, meta = stitch_context(source)
    assert context == " Alpha\ntext \n\n beta "
    for row, span in zip(rows, meta["sources"]):
        assert context[span["start"]:span["end"]] == row["content"]
    assert meta["quality_status"] == {"failed": 1, "passed": 1}


def test_judge_receives_reference_but_no_method_or_previous_grade():
    item = fixture("constructed_response")
    messages = judge_messages(item, "ANONYMOUS_CANDIDATE")
    payload = json.loads(messages[1]["content"])
    assert payload["reference"]["reference_solution"] == "SECRET_REFERENCE"
    assert payload["candidate_answer"] == "ANONYMOUS_CANDIDATE"
    assert "method" not in payload and "grades" not in payload and "id" not in payload["question"]
    assert "SECRET" not in json.dumps(payload["question"])


def test_judge_rejects_missing_duplicate_units_fractional_fill_and_boolean_credit():
    fill = calibration_cases()[0][1]
    for invalid in (grade({"answer": .5}), grade({"answer": True}), grade({"other": 1})):
        with pytest.raises(ValueError):
            validate_grade(fill, invalid)
    item = fixture("constructed_response")
    duplicate = grade({"i": 1})
    duplicate["units"] *= 2
    with pytest.raises(ValueError):
        validate_grade(item, duplicate)


def test_median_is_per_subquestion_and_disagreement_requires_third():
    item = fixture("constructed_response")
    item["subquestions"] += [{"id": "ii", "question": "证明。", "answer": "irrelevant"}]
    first, second, third = grade({"i": 1, "ii": 0}), grade({"i": .5, "ii": 1}), grade({"i": 1, "ii": .5})
    with pytest.raises(ValueError, match="third"):
        aggregate_grades(item, [first, second])
    result = aggregate_grades(item, [first, second, third])
    assert result["unit_credits"] == {"i": 1, "ii": .5}
    assert result["score"] == 4.5 and result["strict_score"] == 0 and not result["fully_correct"]


def test_missing_or_duplicate_predictions_cannot_be_reported_complete(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"id":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        unique_rows(path, ["one", "two"])
    path.write_text('{"id":"one"}\n{"id":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        unique_rows(path)


def test_input_budget_counts_rendered_chat_not_json_escapes():
    class Client:
        config = {"enable_thinking": False}

        def post(self, route, body):
            assert route == "/apply-template" and body["add_generation_prompt"]
            return {"prompt": "CHAT:" + body["messages"][0]["content"]}

        def tokenize(self, text):
            assert text == "CHAT:line one\nline two\\formula"
            return list(text)

    assert count_prompt_tokens(Client(), [{"role": "user", "content": "line one\nline two\\formula"}]) == 30


def test_verbal_judge_rating_maps_to_credit_without_float_generation():
    item = calibration_cases()[3][1]
    value = grade({"1": 0})
    value["units"][0].pop("credit")
    value["units"][0]["rating"] = "incorrect"

    class Client:
        def complete(self, messages, limit, **kwargs):
            unit_schema = kwargs["schema"]["properties"]["units"]["items"]["properties"]
            assert "rating" in unit_schema and "credit" not in unit_schema
            return json.dumps(value)

    assert grade_once(item, "invalid proof", Client(), 1024, "test")["units"][0]["credit"] == 0
