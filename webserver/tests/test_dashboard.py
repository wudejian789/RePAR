"""Browser-level integration checks against the real frozen experiment export."""
from __future__ import annotations

import json
import os
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest

from webserver.server import Handler, WEBROOT


@pytest.fixture(scope="module")
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=WEBROOT))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    worker.join()


@pytest.fixture(scope="module")
def browser():
    sync_playwright = pytest.importorskip("playwright.sync_api", reason="Install Playwright to run browser integration tests").sync_playwright
    candidates = [os.environ.get("REPAR_BROWSER", ""),
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
    executable = next((p for p in candidates if p and Path(p).exists()), None)
    with sync_playwright() as p:
        if executable is None and not Path(p.chromium.executable_path).exists():
            pytest.skip("Set REPAR_BROWSER or run python -m playwright install chromium")
        instance = p.chromium.launch(executable_path=executable, headless=True)
        yield instance
        instance.close()


@pytest.mark.parametrize(("dataset_path", "case_id", "left", "right", "caches"), [
    ("", "66f94f9ebb02136c067c4fde", "direct_27b", "bm25",
     ("baselines_27b_20261119/direct/cache/solver", "cache/llm")),
    ("/math", "MATH_2025_NATIONAL_I_Q17", "direct_08b", "repar_r27_27b",
     ("math_2025_national_i_v5/cache/solver", "repar_v4_r27_27b_20261120/cache/solver")),
])
def test_final_model_input_matches_frozen_request(site, browser, dataset_path, case_id, left, right, caches, tmp_path):
    from playwright.sync_api import expect

    case = json.loads((WEBROOT / "data" / dataset_path.lstrip("/") / "cases" / f"{case_id}.json").read_bytes())
    page = browser.new_page(viewport={"width": 390, "height": 844} if dataset_path else {"width": 1440, "height": 1000})
    requests = []
    page.on("request", lambda request: requests.append(request.url) if "/final-inputs/" in request.url else None)
    query = "?dataset=math" if dataset_path else ""
    page.goto(f"{site}/{query}#case={case_id}&x={left}&y={right}")
    expect(page.locator("[data-final-input]")).to_have_count(2)
    assert not requests

    for method, cache in zip((left, right), caches):
        card = page.locator(f'[data-method-card="{method}"]')
        panel = card.locator("[data-final-input]")
        expect(panel.locator("summary")).to_have_text("最终模型输入")
        assert panel.evaluate("node => node.nextElementSibling.querySelector('summary').textContent") == "模型原始回答"
        panel.locator("summary").click()
        expect(panel).to_have_attribute("data-status", "loaded")
        key = case["methods"][method]["calls"][-1]["key"]
        record = json.loads((WEBROOT.parent / "artifacts" / cache / f"{key}.json").read_bytes())
        expected = record["request"]["messages"]
        assert panel.locator(".input-role").all_text_contents() == [message["role"] for message in expected]
        assert panel.locator("pre").all_text_contents() == [message["content"] for message in expected]

    assert len(requests) == 2
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.locator(".method-comparison").screenshot(path=str(tmp_path / "final-model-input.png"))
    page.close()


def test_real_cases_plot_comparison_evidence_and_trace(site, browser):
    from playwright.sync_api import expect
    overview = json.loads((WEBROOT/"data/overview.json").read_text(encoding="utf-8"))
    page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.goto(site)
    expect(page.locator(".plot-point")).to_have_count(125)
    expect(page.locator(".method-card")).to_have_count(2)
    assert page.locator("#case-list").evaluate("node => node.scrollHeight > node.clientHeight")
    chart = page.locator(".chart-area").bounding_box()
    listing = page.locator(".case-list").bounding_box()
    assert abs(chart["height"] - listing["height"]) < 1
    assert "20.8" in page.locator("#summary").inner_text()
    default_y=overview.get("default_y","repar")
    assert f'{overview["overall"][default_y]["accuracy"]:.1f}' in page.locator("#summary").inner_text()
    expect(page.locator("#evidence-reading .evidence-text")).not_to_be_empty()
    page.screenshot(path=str(WEBROOT/"screenshots/dashboard.png"), full_page=True)
    # Select a different actual scatter point and verify every displayed score.
    selected = overview["cases"][8]
    page.locator(f'.plot-point[data-case="{selected["id"]}"]').click(force=True)
    expect(page.locator(".case-id code")).to_have_text(selected["id"])
    expect(page.locator(".question-text")).to_have_text(selected["question"])
    cards = page.locator(".method-card")
    for i, method in enumerate(("direct", default_y)):
        assert cards.nth(i).get_attribute("data-method-card") == method
        assert f'{selected["methods"][method]["score"]} / 1' in cards.nth(i).inner_text()
    expect(page.locator("[data-evidence-method]")).to_have_count(0)
    expect(page.locator(".evidence-method-label")).to_contain_text(overview["methods"][default_y])
    page.locator("#y-method").select_option("direct_27b")
    expect(page.locator(".evidence-panel")).to_have_count(0)
    expect(page.locator("#knowledge-panel")).to_have_count(0)
    page.locator("#y-method").select_option("direct")
    expect(cards).to_have_count(1)
    # Switch methods, check dense binary scores, and ensure all cases survive.
    page.locator("#x-method").select_option("bm25")
    page.locator("#y-method").select_option("repar_tree")
    assert cards.evaluate_all("(nodes) => nodes.map(node => node.dataset.methodCard)") == ["bm25", "repar_tree"]
    assert page.locator("[data-evidence-method]").evaluate_all(
        "(nodes) => nodes.map(node => node.dataset.evidenceMethod)"
    ) == ["bm25", "repar_tree"]
    page.locator("#metric").select_option("score")
    expect(page.locator(".plot-point")).to_have_count(125)
    expect(page.locator("#log-scale")).to_be_disabled()
    assert "23.2" in page.locator("#summary").inner_text()
    page.locator("#metric").select_option("prompt_tokens")
    page.locator("#log-scale").check()
    expect(page.locator(".plot-point")).to_have_count(125)
    assert "NaN" not in page.locator("#scatter").inner_html()
    assert page.locator(".plot-tick-x").all_text_contents() == page.locator(".plot-tick-y").all_text_contents()
    frame = page.locator(".plot-frame").bounding_box()
    assert abs(frame["width"] - frame["height"]) < 1
    diagonal = page.locator(".plot-diagonal")
    assert float(diagonal.get_attribute("x2")) - float(diagonal.get_attribute("x1")) == (
        float(diagonal.get_attribute("y1")) - float(diagonal.get_attribute("y2"))
    )
    page.locator("#swap-methods").click()
    assert page.locator("#x-method").input_value() == "repar_tree"
    assert page.locator("#y-method").input_value() == "bm25"
    assert cards.evaluate_all("(nodes) => nodes.map(node => node.dataset.methodCard)") == ["repar_tree", "bm25"]
    page.locator("#domain").select_option("Legal")
    expect(page.locator(".plot-point")).to_have_count(14)
    page.locator("#search").fill("no-case-will-match-this-query")
    expect(page.locator(".plot-point")).to_have_count(0)
    expect(page.locator("#case-section")).to_contain_text("没有符合")
    page.locator("#reset-filters").click()
    expect(page.locator(".plot-point")).to_have_count(125)
    page.locator("#x-method").select_option("direct")
    page.locator("#y-method").select_option("repar")
    page.locator("#metric").select_option("service_seconds")
    # Load a trace with a real graph intervention rather than a mock workflow.
    graph_case = "66ec7ba6821e116aacb1d114"
    page.locator("#search").fill(graph_case)
    expect(page.locator(".plot-point")).to_have_count(1)
    expect(page.locator(".case-id code")).to_have_text(graph_case)
    page.locator('[data-open-trace="repar"]').click()
    expect(page.locator("#trace-dialog")).to_be_visible()
    expect(page.locator(".round-node")).to_have_count(6)
    page.locator('[data-round="3"]').click()
    expect(page.locator("#trace-graph")).to_contain_text("graph_search")
    page.locator(".flow-node.evidence").first.click()
    expect(page.locator("#trace-source .evidence-text")).not_to_be_empty()
    page.screenshot(path=str(WEBROOT/"screenshots/trace.png"))
    page.locator('[data-round="4"]').click()
    expect(page.locator(".tool-call.error")).to_have_count(1)
    expect(page.locator("#trace-graph")).to_contain_text("无效节点 ID")
    page.locator('[data-round="5"]').click()
    expect(page.locator(".trace-step")).to_contain_text("标准答案")
    with page.expect_download() as downloaded:
        page.locator("#download-trace").click()
    assert "trace.json" in downloaded.value.suggested_filename
    page.keyboard.press("Escape")
    expect(page.locator("#trace-dialog")).not_to_be_visible()
    # Original evidence is accessible independently of the trace viewer.
    page.locator("#x-method").select_option("bm25")
    page.locator('[data-evidence-method="bm25"]').click()
    expect(page.locator("#evidence-index .evidence-item")).to_have_count(24)
    page.locator("#evidence-search").fill("impossible-evidence-match")
    expect(page.locator("#evidence-index .evidence-item")).to_have_count(0)
    page.locator("#evidence-search").fill("")
    expect(page.locator("#evidence-index .evidence-item")).to_have_count(24)
    with page.expect_download() as downloaded:
        page.locator("#export-csv").click()
    assert "1-cases.csv" in downloaded.value.suggested_filename
    assert not errors, errors
    page.close()


def test_fallback_and_graph_disabled_trajectories(site, browser):
    from playwright.sync_api import expect
    overview = json.loads((WEBROOT/"data/overview.json").read_text(encoding="utf-8"))
    item = next(row for row in overview["cases"] if row["methods"]["repar"]["fallback"])
    page = browser.new_page()
    page.goto(site+"/#x=repar&y=repar_tree&case="+item["id"])
    expect(page.locator(".case-id code")).to_have_text(item["id"])
    page.locator('[data-open-trace="repar"]').click()
    expect(page.locator(".round-node")).to_have_count(7)
    page.locator('[data-round="5"]').click()
    expect(page.locator(".trace-step")).to_contain_text("回退步骤")
    expect(page.locator(".flow-node.evidence")).to_have_count(12)
    page.locator("#close-trace").click()
    page.locator("#search").fill("66ec7ba6821e116aacb1d114")
    expect(page.locator(".case-id code")).to_have_text("66ec7ba6821e116aacb1d114")
    page.locator('[data-open-trace="repar_tree"]').click()
    page.locator('[data-round="3"]').click()
    expect(page.locator(".tool-error-text")).to_contain_text("disabled operator")
    page.close()


def test_mobile_layout_and_local_file_boundary(site, browser):
    from playwright.sync_api import expect
    page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    page.goto(site+"/#y=repar")
    expect(page.locator(".method-card")).to_have_count(2)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile page overflows horizontally"
    page.locator('[data-open-trace="repar"]').click()
    expect(page.locator("#trace-dialog")).to_be_visible()
    page.screenshot(path=str(WEBROOT/"screenshots/mobile.png"), full_page=True)
    page.locator("#close-trace").click()
    assert page.request.get(site+"/server.py").status == 404
    assert page.request.get(site+"/../README.md").status == 404
    assert page.request.get(site+"/data/").status == 404
    assert page.request.get(site+"/health").ok
    page.close()


def test_exported_scores_and_evidence_match_source_artifacts():
    from repar.io import read_json, read_jsonl
    from webserver.frozen_data import RUNS
    root = WEBROOT.parent
    overview = read_json(WEBROOT/"data/overview.json")
    data = {r["_id"]: r for r in read_json(root/"data/longbench_v2.json") if r["domain"] == "Multi-Document QA"}
    results = {m: {r["_id"]:r for r in read_jsonl(root / ("artifacts/experiment_v2" if m.endswith("_v2") else "artifacts/experiment") / ((m.removesuffix("_v2"))+".jsonl"))} for m in overview["methods"] if m not in RUNS}
    for method, (_, directory) in RUNS.items():
        with (root / directory / "longbench/predictions.jsonl").open(encoding="utf-8") as stream:
            results[method] = {row["id"]: row for row in map(json.loads, stream)}
    assert len(overview["cases"]) == len(data) == 125
    evidence_count = 0
    for row in overview["cases"]:
        case = read_json(WEBROOT/f'data/cases/{row["id"]}.json')
        original = data[row["id"]]
        assert case["question"] == original["question"] and case["answer"] == original["answer"]
        for method, value in case["methods"].items():
            source = results[method][row["id"]]
            assert value["pred"] == source["pred"]
            assert value["score"] == int(source["correct"] if method in RUNS else source["judge"])
            assert value["evidence_ids"] == source.get("evidence_ids", [])
            records = value.get("evidence", case["evidence"])
            assert set(value["evidence_ids"]) <= set(records)
            assert set(value["retrieved_ids"]) <= set(records)
            if method in RUNS:
                assert value["evidence_records"] == source["evidence_records"]
                for record in value["evidence_records"] + value["retrieved_records"]:
                    assert record["text"] == original["context"][record["start"]:record["end"]]
        for evidence in case["evidence"].values():
            assert evidence["text"] == original["context"][evidence["start"]:evidence["end"]]
            evidence_count += 1
    assert evidence_count > 3000


def test_v2_phase_order_entity_graph_and_source_mapping(site,browser):
    from playwright.sync_api import expect
    overview=json.loads((WEBROOT/"data/overview.json").read_text(encoding="utf-8"))
    if "repar_v2" not in overview["methods"]:
        pytest.skip("Full v2 export is not ready")
    chosen=None
    for item in overview["cases"]:
        trace=json.loads((WEBROOT/f'data/traces/repar_v2/{item["id"]}.json').read_text(encoding="utf-8"))
        for i,step in enumerate(trace["trace"]):
            if any(c.get("name")=="graph_neighbors" and c.get("observation",{}).get("relations") for c in step.get("executed",[])):
                chosen=(item,i,trace);break
        if chosen:break
    assert chosen,"Expected at least one real entity-neighbor traversal"
    item,round_index,trace=chosen
    page=browser.new_page(viewport={"width":1440,"height":1050})
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto(site+"/#case="+item["id"]+"&x=direct&y=repar_v2")
    expect(page.locator(".method-card")).to_have_count(2)
    expect(page.locator(".entity-knowledge")).to_be_visible()
    page.locator(".entity-knowledge summary").first.click()
    expect(page.locator(".entity-pill").first).to_be_visible()
    page.locator('[data-open-trace="repar_v2"]').click()
    page.locator(f'[data-round="{round_index}"]').click()
    expect(page.locator(".phase-sequence")).to_contain_text("1 PLAN")
    details=page.locator(".trace-step-content").inner_text()
    assert details.index("PLAN / 执行前计划")<details.index("REFLECTION / 执行后证据评估")
    page.locator('[data-graph-view="entities"]').click()
    expect(page.locator(".flow-node.entity").first).to_be_visible()
    expect(page.locator(".entity-edge").first).to_be_visible()
    page.locator(".flow-node.entity").first.click()
    expect(page.locator("#trace-source")).to_contain_text("个原文片段")
    page.screenshot(path=str(WEBROOT/"screenshots/entity_graph_v2.png"))
    page.locator('[data-graph-view="flow"]').click()
    expect(page.locator(".flow-node.tool")).to_contain_text("graph_neighbors")
    page.locator("#close-trace").click()
    assert not errors,errors
    page.close()
