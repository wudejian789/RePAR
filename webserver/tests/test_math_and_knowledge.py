"""Checks for frozen math marks and provenance navigation."""
import json

from webserver.server import WEBROOT
from webserver.tests.test_dashboard import browser, site  # noqa: F401
from webserver.frozen_data import HISTORICAL_MATH_RUN, MATH_BASELINES, RUNS


def test_math_export_matches_frozen_grades():
    data = WEBROOT / "data/math"
    overview = json.loads((data / "overview.json").read_text(encoding="utf-8"))
    assert len({case["id"] for case in overview["cases"]}) == 19
    grade_files = {method: f"{source}/math/grades.jsonl" for method, (_, source) in RUNS.items()}
    grade_files.update({method: f"{HISTORICAL_MATH_RUN}/{name}_grades.jsonl"
                        for method, (_, name) in MATH_BASELINES.items()})
    assert overview["methods"].keys() == grade_files.keys()
    for method, source in grade_files.items():
        with (WEBROOT.parent / source).open(encoding="utf-8") as stream:
            grades = {row["id"]: row for row in map(json.loads, stream)}
        for item in overview["cases"]:
            cid = item["id"]
            result = item["methods"][method]
            assert result["score"] == grades[cid]["score"]
            assert result["max_score"] == grades[cid]["max_score"]
            case = json.loads((data / "cases" / f"{cid}.json").read_text(encoding="utf-8"))
            assert case["methods"][method]["grade"] == grades[cid]


def test_math_partial_scores_and_reference_are_browsable(site, browser):
    from playwright.sync_api import expect

    page = browser.new_page(viewport={"width": 1430, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(site)
    page.locator("#math-link").click()
    expect(page.locator(".plot-point")).to_have_count(19)
    expect(page.locator(".method-card")).to_have_count(2)
    expect(page.locator("#y-method")).to_have_value("repar_v5_08b")
    expect(page.locator("#summary")).to_contain_text("40.417")
    page.locator('[data-open-trace="repar_v5_08b"]').click()
    draft = page.locator(".trace-draft")
    draft.locator("summary").click()
    expect(draft.locator("pre")).not_to_be_empty()
    page.locator("#close-trace").click()
    expect(page.locator("#trace-dialog")).to_be_hidden()
    page.locator("#x-method").select_option("direct_08b")
    expect(page.locator("#summary .stat-value").first).to_contain_text("18")
    page.locator("#x-method").select_option("bm25_08b")
    expect(page.locator("#summary .stat-value").first).to_contain_text("25")
    page.locator("#y-method").select_option("repar_r27_08b")
    expect(page.locator("#summary")).to_contain_text("40.417")
    page.locator("#domain").select_option("解答题")
    page.locator("#search").fill("MATH_2025_NATIONAL_I_Q17")
    expect(page.locator(".case-id code")).to_have_text("MATH_2025_NATIONAL_I_Q17")
    page.locator("#reference-solution summary").click()
    expect(page.locator("#reference-solution pre")).to_contain_text("√2/3")
    page.locator('[data-evidence-method="bm25_08b"]').click()
    expect(page.locator("#evidence-index .evidence-item")).to_have_count(24)
    page.locator("#y-method").select_option("repar_r27_27b")
    expect(page.locator("#reference-solution")).to_have_attribute("open", "")
    expect(page.locator('[data-evidence-method="bm25_08b"]')).to_have_attribute("aria-pressed", "true")
    score = page.locator(".method-card").filter(has_text="RePAR v4-r27 · 27B")
    score.get_by_text("模型原始回答").click()
    expect(score.locator("pre")).to_contain_text(r"\sqrt")
    score.get_by_text("评分记录").click()
    score.get_by_text("评审 1", exact=True).click()
    expect(score.locator(".judge-unit").first).to_contain_text("小问 1")
    score.locator('[data-open-trace="repar_r27_27b"]').click()
    expect(page.locator("#trace-dialog")).to_be_visible()
    expect(page.locator(".trace-step")).to_contain_text("执行后")
    page.locator(".final-node").click()
    expect(page.locator(".trace-step")).to_contain_text("实验分 15 / 15")
    page.keyboard.press("Escape")
    page.reload()
    expect(page.locator(".case-id code")).to_have_text("MATH_2025_NATIONAL_I_Q17")
    expect(page.locator("#x-method")).to_have_value("bm25_08b")
    assert not errors
    page.close()


def test_knowledge_tree_graph_and_bm25_source_navigation(site, browser):
    from playwright.sync_api import expect

    item_id = "66f3cca6821e116aacb2ef7c"
    item = json.loads((WEBROOT / "data/cases" / f"{item_id}.json").read_text(encoding="utf-8"))
    page = browser.new_page(viewport={"width": 1440, "height": 1050})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{site}/#case={item_id}&x=bm25_27b&y=repar_r27_27b")
    page.locator("#knowledge-panel > summary").click()
    expect(page.locator("#knowledge-method option")).to_have_count(2)
    expect(page.locator(".tree-root")).to_contain_text("T0")
    expect(page.locator(".knowledge-graph svg")).to_be_visible()
    child = page.locator(".tree-children button").first
    child_id = child.get_attribute("data-knowledge-tree")
    child.click()
    expect(page.locator(".tree-root")).to_contain_text(child_id)
    page.locator("#knowledge-final").click()
    first = item["methods"]["repar_r27_27b"]["evidence_ids"][0]
    expect(page.locator(".knowledge-detail h3").first).to_contain_text(first)
    page.locator(".knowledge-pane button[data-knowledge-edge]").first.click()
    expect(page.locator(".knowledge-proposition").first).to_be_visible()
    edge = page.locator(".knowledge-proposition").first
    source = edge.inner_text().split(" · ", 1)[0]
    page.get_by_role("button", name=f"回读 {source}", exact=True).first.click()
    expect(page.locator(".knowledge-detail h3").first).to_contain_text(source)
    page.locator("#knowledge-method").select_option("bm25_27b")
    expect(page.locator(".knowledge-pane h3").first).to_have_text("BM25 原文块")
    page.locator("#knowledge-search").fill("Precautionary Principle")
    page.locator('#knowledge-search-results [data-knowledge-chunk="C3"]').click()
    expect(page.locator("#knowledge-workspace .evidence-text")).to_contain_text("Precautionary Principle")
    expect(page.locator(".knowledge-graph")).to_have_count(0)
    assert not errors
    page.close()
