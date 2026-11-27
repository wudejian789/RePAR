from repar.benchmarks.math_exam import formatted_solver_messages, linear_math_notation, task_text


def test_linear_notation_preserves_coefficients_and_power_boundaries():
    assert linear_math_notation("v²/a² = 2√7/5; g = 12√(v−3)") == "v^(2)/a^(2) = 2*sqrt(7)/5; g = 12*sqrt(v-3)"


def test_linear_question_preserves_direction_and_original_source():
    item = {"question_stem": "求有向线段的长度。", "question_type": "fill_in_blank", "choices": [],
            "figures": [{"description": "有向线段。", "unit": "m",
                         "vectors": [{"label": "位移", "start": [2, 3], "end": [-5, 1]}]}]}
    evidence = "Source equation x² = 7."
    body = formatted_solver_messages(task_text(item), evidence, "", linear_notation=True)[1]["content"]
    assert "起点 (2, 3)，终点 (-5, 1)" in body
    assert body.endswith(evidence)
