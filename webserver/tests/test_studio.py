"""Browser regression for per-turn sources and knowledge management."""

from datetime import datetime, timezone
import json
from urllib.parse import parse_qs, urlsplit

from webserver.tests.test_dashboard import browser, site  # noqa: F401


def test_manage_library_and_switch_libraries_per_turn(site, browser):
    from playwright.sync_api import expect

    source = "The archive is stored in the north depot."
    other_source = "The original archive moved to the east wing."
    original = {"id": "a" * 32, "name": "Archive guide", "status": "ready", "sources": []}
    library = {
        "id": "b" * 32, "name": "Team notes", "status": "ready",
        "sources": [{"name": "notes.txt", "start": 0, "end": len(source)}],
        "stats": {"leaves": 2, "entities": 0, "relations": 0},
    }
    session = {"id": "c" * 32, "title": "New chat", "run_count": 0, "auto_title": True}
    runs = [
        {"id": "d" * 32, "session_id": session["id"], "library_id": library["id"],
         "task": "Where is the archive?", "status": "ready", "response": "North depot.",
         "evidence_ids": ["C0"]},
        {"id": "e" * 32, "session_id": session["id"], "library_id": original["id"],
         "task": "Where did it move?", "status": "ready", "response": "East wing.",
         "evidence_ids": ["C0"]},
    ]
    texts = {library["id"]: source, original["id"]: other_source}
    library_list = [original]
    submissions = {"runs": []}
    removed = False

    def route_api(route):
        nonlocal removed
        path = route.request.url.split("/api/studio/")[1].split("?")[0]
        if path == f"sessions/{session['id']}" and route.request.method == "DELETE":
            removed = True
            return route.fulfill(json={"id": session["id"]})
        if route.request.method == "POST":
            if path == "libraries":
                submissions[path] = route.request.post_data_json
                library_list.insert(0, library)
                return route.fulfill(status=202, json=library)
            if path == "sessions":
                submissions[path] = route.request.post_data_json
                return route.fulfill(status=202, json=session)
            if path == "runs":
                run = runs[len(submissions[path])]
                submissions[path].append(route.request.post_data_json)
                return route.fulfill(status=202, json=run)
        if path == "libraries":
            return route.fulfill(json={"libraries": library_list})
        if path == "sessions":
            saved = {**session, "last_library_id": original["id"], "run_count": len(runs)}
            return route.fulfill(json={"sessions": [saved] if not removed and len(submissions["runs"]) == 2 else []})
        if path.startswith("sessions/"):
            return route.fulfill(json={"runs": runs[::-1]})
        if path.endswith("/index"):
            text = texts[path.split("/")[1]]
            return route.fulfill(json={
                "kind": "entity", "nodes": [], "propositions": [], "entities": [], "relations": [],
                "chunks": [{"id": "C0", "start": 0, "end": len(text), "text": text, "entity_ids": []}],
            })
        if path.startswith("runs/"):
            run = next(run for run in runs if run["id"] == path.split("/")[1])
            if path.endswith("/trace"):
                return route.fulfill(json={"trace": [], "calls": [], "state": {}, "response": run["response"]})
            text = texts[run["library_id"]]
            observation = {"stage": "observation", "round": 1, "executed": [{
                "name": "read", "observation": {"status": "ok"}, "tool_seconds": 0.01,
                "visible_observation": {"source_views": [
                    {"id": "C0", "start": 0, "end": 11, "text": text[:11]},
                    {"id": "C0", "start": 12, "end": len(text), "text": text[12:]},
                ]},
            }]}
            return route.fulfill(json={"job": run, "offset": 1, "events": [observation]})
        return route.fulfill(status=404, json={"error": path})

    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("**/api/studio/**", route_api)
    try:
        page.goto(f"{site}/studio/libraries")
        expect(page.locator("#library-manager")).to_be_visible()
        page.locator("#new-library").click()
        page.locator("#files").set_input_files({
            "name": "notes.txt", "mimeType": "text/plain", "buffer": source.encode("utf-8"),
        })
        page.locator("#library-name").fill(library["name"])
        page.locator("#build").click()
        expect(page.locator("#upload-form")).to_be_hidden()
        expect(page.locator("#library-details")).to_contain_text("notes.txt")
        expect(page.locator(".session-panel")).to_be_hidden()
        details_tab = page.locator("#library-tab-details")
        index_tab = page.locator("#library-tab-index")
        expect(details_tab).to_have_attribute("aria-selected", "true")
        index_tab.click()
        expect(page.locator("#index-host .knowledge-pane").first).to_be_visible()
        expect(page.locator("#library-summary")).to_be_hidden()
        expect(index_tab).to_have_attribute("aria-selected", "true")
        assert "/studio/libraries?" in page.url
        page.locator("#ask-with-library").click()
        expect(page.locator(".chat-welcome")).to_be_visible()
        expect(page.locator("#session-library")).to_have_value(library["id"])
        page.locator(".starter-question").first.click()
        assert page.locator("#question").input_value()
        page.locator("#question").fill(runs[0]["task"])
        page.locator("#ask").click()
        expect(page.locator(".message-scope")).to_have_text("Team notes")
        page.locator("#session-library").select_option(original["id"])
        page.locator("#follow-up").check()
        page.locator("#question").fill(runs[1]["task"])
        page.locator("#ask").click()
        expect(page.locator(".message-scope")).to_have_text(["Team notes", "Archive guide"])
        assert submissions == {
            "libraries": {"name": "Team notes", "documents": [{"name": "notes.txt", "text": source}]},
            "sessions": {},
            "runs": [
                {"library_id": library["id"], "session_id": session["id"], "task": runs[0]["task"]},
                {"library_id": original["id"], "session_id": session["id"], "task": runs[1]["task"],
                 "previous_run": runs[0]["id"]},
            ],
        }
        expect(page.locator(".session-option")).to_contain_text("2 轮问答")
        page.reload()
        expect(page.locator("#session-library")).to_have_value(original["id"])
        expect(page.locator(".message-scope")).to_have_text(["Team notes", "Archive guide"])
        expect(page.locator("#trace-panel")).to_be_visible()
        expect(page.locator("#trace-question")).to_have_text(runs[1]["task"])
        citations = page.locator(".reply-actions .source-pill")
        for index, text in enumerate((source, other_source)):
            citations.nth(index).click()
            expect(page.locator(".reply-source .source-quotation")).to_have_text(text)
            expect(page.get_by_role("link", name="在知识库中查看")).to_have_attribute(
                "href", f"/studio/libraries?library={runs[index]['library_id']}&block=C0",
            )
            expect(citations.nth(index)).to_have_attribute("aria-expanded", "true")
            citations.nth(index).click()
            expect(page.locator(".reply-source .source-quotation")).to_have_count(0)
        page.get_by_role("button", name="查看轨迹").first.click()
        expect(page.locator("#trace-question")).to_have_text(runs[0]["task"])
        page.locator(".observation-tool-title").click()
        source_heading = page.locator(".observation-sources > summary")
        source_heading.click()
        fragments = page.locator(".observation-fragment")
        expect(fragments.locator("summary")).to_have_text([
            "C0 · 字符 0–11", f"C0 · 字符 12–{len(source)}",
        ])
        assert fragments.first.locator("summary").bounding_box()["x"] - source_heading.bounding_box()["x"] >= 16
        for fragment in fragments.all():
            fragment.locator("summary").click()
            expect(fragment.locator("pre")).to_be_visible()
        expect(fragments.locator("pre")).to_have_text([source[:11], source[12:]])
        expect(page.locator("dialog[open]")).to_have_count(0)
        fragments.first.locator("summary").click()
        expect(fragments.first.locator("pre")).to_be_hidden()
        expect(fragments.last.locator("pre")).to_be_visible()
        page.locator("#raw-trace").click()
        expect(page.locator("#download")).to_be_enabled()
        page.get_by_text("事件 JSON · 原始调试记录", exact=True).click()
        expect(page.locator("#raw-content details[open] pre").last).to_contain_text('"stage": "observation"')
        with page.expect_download() as download:
            page.locator("#download").click()
        assert download.value.suggested_filename == f"repar-v5-{runs[0]['id']}.json"
        page.locator("#raw-trace").click()
        expect(page.locator("#raw-content")).to_be_hidden()
        expect(page.locator("#timeline")).to_be_visible()
        page.locator("#new-session").click()
        expect(page.locator(".chat-welcome")).to_be_visible()
        expect(page.locator("#trace-panel")).to_be_hidden()
        page.locator(".delete-session").click()
        expect(page.locator("#delete-session-dialog")).to_be_visible()
        page.locator("#cancel-session-delete").click()
        assert not removed
        page.locator(".session-option").click()
        expect(page.locator(".exchange")).to_have_count(2)
        page.locator(".delete-session").click()
        page.locator("#confirm-session-delete").click()
        expect(page.locator("#delete-session-dialog")).to_be_hidden()
        expect(page.locator(".session-option")).to_have_count(0)
        expect(page.locator(".chat-welcome")).to_be_visible()
        expect(page.locator("#trace-panel")).to_be_hidden()
        assert removed
        page.reload()
        expect(page.locator(".session-option")).to_have_count(0)
        expect(page.locator("#question")).to_be_enabled()
        assert not errors
    finally:
        page.close()


def test_running_reply_and_inline_trace_update_independently(site, browser):
    from playwright.sync_api import expect

    library = {"id": "library", "name": "Field notes", "status": "ready", "sources": []}
    session = {"id": "session", "title": "Field notes QA", "run_count": 0, "auto_title": True}
    job = {"id": "run", "session_id": session["id"], "library_id": library["id"],
           "task": r"Which trial used the blue marker? Check $\sqrt{3} \approx 1.73205081$.", "status": "queued",
           "created": datetime.now(timezone.utc).isoformat()}
    events = [{"stage": "direct_answer", "answer": "Initially unclear."}]
    plan = {"stage": "plan", "round": 1, "plan": {"objective": "Find the marker in the notes", "query": "blue marker"}}
    observation = {"stage": "observation", "round": 1, "executed": [{
        "name": "tree_search", "tool_seconds": 0.04, "parameters": {"query": "blue marker"},
        "observation": {"status": "ok", "results": [{"id": "C0", "text": "Trial A used the blue marker. " * 250}]},
        "visible_observation": {"source_views": [{"id": "C0", "text": "Trial A used the blue marker. " * 250}]},
    }]}
    offline = False
    usage = {"calls": 0, "measured": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def route_api(route):
        url = urlsplit(route.request.url)
        path = url.path.removeprefix("/api/studio/")
        if path == "libraries":
            return route.fulfill(json={"libraries": [library]})
        if path == "sessions":
            return route.fulfill(json=session if route.request.method == "POST" else {"sessions": []})
        if path == "runs":
            return route.fulfill(status=202, json={**job, "status": "queued"})
        if path == "runs/run/calls/plan/1":
            return route.fulfill(json={"calls": [{
                "tag": "v4_plan_1", "request": {"temperature": 0, "messages": [
                    {"role": "system", "content": "Use the field notes.\nPreserve source identifiers."},
                    {"role": "user", "content": job["task"]},
                ]},
                "response": {"choices": [{"message": {"content": "Find the marker"}}],
                             "usage": {"prompt_tokens": 91, "completion_tokens": 5}},
                "seconds": 0.12,
            }, {
                "tag": "v4_plan_1_repair", "request": {"messages": [{"role": "user", "content": "Repair the format."}]},
                "response": {"choices": [{"message": {"content": "Repaired output"}}]},
                "cached": True,
            }]})
        if path == "runs/run":
            if offline:
                return route.abort()
            offset = int(parse_qs(url.query)["offset"][0])
            return route.fulfill(json={"job": job, "usage": usage, "events": events[offset:] if job["status"] != "queued" else [],
                                       "offset": len(events) if job["status"] != "queued" else 0})
        return route.fulfill(status=404, json={"error": path})

    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("**/api/studio/**", route_api)
    try:
        page.goto(f"{site}/studio")
        page.locator("#question").fill(job["task"])
        page.locator("#ask").click()
        stage = page.locator(".run-stage")
        timeline = page.locator("#timeline")
        detail = page.locator("#event-detail")
        expect(stage).to_have_text("已排队")
        expect(page.locator("#trace-usage")).to_be_hidden()
        expect(page.locator(".question-message math msqrt")).to_have_count(1)
        expect(page.locator("#trace-panel")).to_be_visible()
        expect(page.locator("#trace-panel")).to_be_in_viewport()
        assert page.locator("#trace-panel").bounding_box()["x"] > page.locator("#messages").bounding_box()["x"]
        expect(page.locator("#trace-question")).to_have_text(job["task"])
        expect(page.locator("#question")).to_be_enabled()
        elapsed = page.locator(".run-elapsed")
        start_label = elapsed.inner_text()
        expect(elapsed).not_to_have_text(start_label)
        job["status"] = "running"
        usage.update(calls=1, measured=1, prompt_tokens=90, completion_tokens=0)
        expect(stage).to_have_text("已生成初步回答")
        expect(page.locator("#trace-usage")).to_have_text("输入 90 · 输出 0 tokens · 1 次调用")
        expect(timeline.locator("button")).to_have_count(1)
        offline = True
        expect(page.locator(".run-stage-detail")).to_contain_text("正在重连")
        offline = False
        events.extend([plan, observation])
        usage.update(calls=3, measured=2, prompt_tokens=181, completion_tokens=5)
        expect(stage).to_have_text("第 1 轮 · 已收到查询结果")
        expect(page.locator("#trace-usage")).to_have_text("输入 181 · 输出 5 tokens · 已统计 2/3 次调用")
        expect(timeline.locator('[data-stage="observation"]')).to_have_attribute("aria-pressed", "true")
        expect(page.locator(".run-stage-detail")).to_contain_text("1 段原文")
        box = timeline.bounding_box()
        tool = detail.locator(".observation-tool")
        heading = tool.locator(".observation-tool-title")
        heading.click()
        sections = tool.locator(".observation-tool-content > details")
        expect(sections.locator(":scope > summary")).to_have_text(
            ["工具原始输入", "工具完整输出", "提交给 Reflection 的观察", "原文出处 · 1 段"],
        )
        assert sections.first.bounding_box()["x"] > heading.bounding_box()["x"]
        execution = observation["executed"][0]
        for index, key in ((0, "parameters"), (2, "visible_observation"), (1, "observation")):
            section = sections.nth(index)
            section.locator("summary").click()
            expect(section.locator("pre")).not_to_be_empty()
            assert json.loads(section.locator("pre").text_content()) == execution[key]
            if key != "observation":
                section.locator("summary").click()
        expect(detail.get_by_text("事件 JSON", exact=False)).to_have_count(0)
        assert timeline.bounding_box() == box
        assert detail.evaluate("el => el.scrollHeight > el.clientHeight")
        expect(stage).to_be_visible()
        timeline.locator('[data-stage="plan"]').click()
        expect(detail.locator("pre")).to_have_count(0)
        detail.get_by_text("模型原始输入与输出", exact=True).click()
        expect(detail.locator(".call-message h3")).to_have_text(["system", "user"])
        expect(detail.locator(".call-usage")).to_have_text("输入 91 · 输出 5 tokens · 0.12 s")
        assert detail.locator(".call-message pre").first.text_content() == "Use the field notes.\nPreserve source identifiers."
        assert detail.locator(".call-message pre").last.text_content() == job["task"]
        expect(detail).not_to_contain_text("temperature")
        expect(detail).not_to_contain_text("v4_plan_1")
        detail.get_by_role("button", name="输出正文").click()
        expect(detail.locator(".call-content pre")).to_have_text("Find the marker")
        detail.get_by_role("combobox", name="调用次数").select_option("1")
        expect(detail.locator(".call-content pre")).to_have_text("Repaired output")
        expect(detail.locator(".call-usage")).to_have_text("Token 用量未记录 · 缓存复用")
        detail.get_by_role("combobox", name="调用次数").select_option("0")
        detail.get_by_role("button", name="JSON").click()
        expect(detail.get_by_text("请求 JSON", exact=True)).to_be_visible()
        expect(detail.locator(".call-content pre").first).to_contain_text('"temperature": 0')
        events.append({"stage": "finalizing", "termination": "model_sufficient"})
        expect(stage).to_have_text("正在生成最终回答")
        expect(timeline.locator('[data-stage="plan"]')).to_have_attribute("aria-pressed", "true")
        page.locator("#follow-trace").click()
        expect(timeline.locator('[data-stage="finalizing"]')).to_have_attribute("aria-pressed", "true")
        job.update(status="ready", seconds=8.0, response=(
            r"Trial A. $\sqrt{3} \approx 1.73205081$." + "\n" + r"$$\frac{1}{2}$$" + "\n"
            + "\n".join("The marker was recorded with the rest of the observations." for _ in range(35))
        ))
        expect(page.locator(".reply-body math msqrt")).to_have_count(1)
        expect(page.locator('.reply-body math[display="block"] mfrac')).to_have_count(1)
        expand = page.locator(".reply-expand")
        expect(expand).to_be_visible()
        expect(expand).to_have_attribute("aria-expanded", "false")
        preview = page.locator(".reply-preview")
        assert preview.evaluate("el => el.scrollHeight > el.clientHeight")
        expand.click()
        expect(expand).to_have_attribute("aria-expanded", "true")
        assert preview.evaluate("el => el.scrollHeight === el.clientHeight")
        expand.click()
        expect(expand).to_have_attribute("aria-expanded", "false")
        expect(stage).to_have_count(0)
        expect(page.locator("#ask")).to_be_enabled()
        assert not errors
    finally:
        page.close()


def test_graph_lives_in_management_and_links_original_blocks(site, browser):
    from playwright.sync_api import expect

    source = "The Skylark archive is located in the north depot."
    library = {"id": "archive", "name": "Archive", "status": "ready",
               "sources": [{"name": "archive.txt", "start": 0, "end": len(source)}]}
    other = {"id": "handbook", "name": "Handbook", "status": "ready", "sources": []}
    extra = "A separate passage about a maintenance schedule."
    index = {
        "kind": "entity", "nodes": [], "propositions": [],
        "entities": [
            {"id": "E0", "name": "Skylark archive", "type": "ORG", "mentions": [{}], "passage_ids": ["C0"]},
            {"id": "E1", "name": "north depot", "type": "LOC", "mentions": [{}], "passage_ids": ["C0"]},
        ],
        "relations": [{
            "id": "R0", "subject": "E0", "object": "E1", "predicate": "located in",
            "kind": "predicate", "passage_ids": ["C0"], "proposition_ids": [],
        }],
        "chunks": [
            {"id": "C0", "start": 0, "end": len(source), "text": source, "entity_ids": ["E0", "E1"]},
            {"id": "C1", "start": len(source)+2, "end": len(source)+2+len(extra), "text": extra, "entity_ids": []},
        ],
    }
    responses = {
        "libraries": {"libraries": [library, other]},
        "libraries/archive/index": index,
        "libraries/handbook/index": {**index, "chunks": index["chunks"][:1]},
    }
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("**/api/studio/**", lambda route: route.fulfill(
        json=responses[route.request.url.split("/api/studio/")[1].split("?")[0]],
    ))
    try:
        page.goto(f"{site}/studio/libraries")
        expect(page.locator("#chat-view")).to_be_hidden()
        expect(page.locator("#library-details h2")).to_have_text("Archive")
        graph_tab = page.locator("#library-tab-graph")
        details_tab = page.locator("#library-tab-details")
        index_tab = page.locator("#library-tab-index")
        header = page.locator("#library-heading")
        initial_bounds = header.bounding_box()
        expect(details_tab).to_have_attribute("aria-selected", "true")
        expect(graph_tab).to_have_attribute("aria-selected", "false")
        graph_tab.click()
        expect(graph_tab).to_have_attribute("aria-selected", "true")
        expect(page.locator("#library-directory")).to_be_visible()
        expect(page.locator("#library-details h2")).to_have_text("Archive")
        expect(page.locator("#library-summary")).to_be_hidden()
        expect(page.locator("#ask-with-library")).to_be_visible()
        assert header.bounding_box() == initial_bounds
        graph = page.locator("#knowledge-graph")
        search = page.locator("#kg-search")
        blocks = page.locator("#graph-block-list .block-option")
        expect(blocks).to_have_count(2)
        assert graph.evaluate("host => host._cyreg.cy.nodes(':visible').length") == 3

        search.fill("xyzzyqz")
        assert graph.evaluate("host => host._cyreg.cy.nodes(':visible').length") == 3
        assert graph.evaluate("host => host._cyreg.cy.elements('.context-dim').length") == 0
        graph.evaluate("host => { host._cyreg.cy.getElementById('R0').emit('tap'); }")
        expect(page.locator(".detail-record-id")).to_have_text("R0")
        expect(blocks.first).to_have_attribute("aria-pressed", "true")
        assert graph.evaluate("host => host._cyreg.cy.getElementById('C0').hasClass('block-match')")
        graph_bounds = graph.bounding_box()
        page.locator(".detail-source-link button").click()
        expect(page.locator("#graph-source .source-quotation")).to_have_text(source)
        assert graph.bounding_box() == graph_bounds
        graph.evaluate("""host => {
            const cy = host._cyreg.cy;
            cy.stop().zoom(0.5);
            host.zoomSamples = [];
            cy.on('zoom', () => host.zoomSamples.push(cy.zoom()));
        }""")
        block_list = page.locator("#graph-block-list")
        list_top = block_list.bounding_box()["y"] - graph.bounding_box()["y"]
        blocks.last.click()
        expect(page.locator("#graph-record")).to_be_hidden()
        expect(page.locator("#graph-source")).to_be_hidden()
        expect(blocks.last).to_have_attribute("aria-pressed", "true")
        assert block_list.bounding_box()["y"] - graph.bounding_box()["y"] == list_top
        assert header.bounding_box() == initial_bounds
        assert graph.evaluate("host => host._cyreg.cy.getElementById('C1').hasClass('block-match')")
        page.wait_for_function("""() => document.getElementById('knowledge-graph')._cyreg.cy.zoom() >= 1.29""")
        assert graph.evaluate("host => host.zoomSamples.some(zoom => zoom > 0.5 && zoom < 1.29)")
        page.locator("#read-graph-block").click()
        expect(page.locator("#graph-source .source-quotation")).to_have_text(extra)
        expect(page.locator("#read-graph-block")).to_have_attribute("aria-expanded", "true")
        page.locator("#read-graph-block").click()
        expect(page.locator("#graph-source")).to_be_hidden()
        page.locator("#graph-reset").click()
        expect(search).to_have_value("")
        expect(page.locator("#graph-record")).to_be_hidden()
        expect(blocks).to_have_count(2)
        page.get_by_role("button", name="Handbook").click()
        expect(page.locator("#library-details h2")).to_have_text("Handbook")
        expect(blocks).to_have_count(1)
        expect(graph_tab).to_have_attribute("aria-selected", "true")
        graph_tab.focus()
        page.keyboard.press("ArrowLeft")
        expect(index_tab).to_be_focused()
        expect(page.locator("#index-host .knowledge-pane").first).to_be_visible()
        expect(graph).to_have_count(0)
        assert header.bounding_box() == initial_bounds
        details_tab.click()
        expect(page.locator("#library-summary")).to_be_visible()
        expect(index_tab).to_have_attribute("aria-selected", "false")
        expect(page.locator("#index-host")).to_be_hidden()
        assert header.bounding_box() == initial_bounds
        page.goto(f"{site}/studio/libraries?library=archive&block=C1")
        expect(blocks.last).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#library-directory")).to_be_visible()
        page.reload()
        expect(blocks.last).to_have_attribute("aria-pressed", "true")
        assert not errors
    finally:
        page.close()
