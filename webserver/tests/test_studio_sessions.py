"""Persistence and migration behavior for chat sessions."""

import numpy as np
import pytest

from repar.dense_index import DenseKnowledgeIndex
from repar.io import append_jsonl, digest, write_json
from webserver.studio import SessionBusyError, Studio, now, scoped_index


def test_block_scope_reuses_vectors_and_excludes_unselected_text(monkeypatch):
    index = DenseKnowledgeIndex.__new__(DenseKnowledgeIndex)
    index.texts = ["first source", "unselected source", "third source"]
    index.context = "\n\n".join(index.texts)
    index.matrix = np.eye(3)
    index.chunk_tokens = [2, 2, 2]
    index.knowledge = [{"concepts": [], "logic_cues": []}] * 3
    index.config = {"tree_fanout": 2}
    index.stats = {"leaves": 3}
    monkeypatch.setattr(DenseKnowledgeIndex, "_build_graph", lambda self: None)
    scoped = scoped_index(index, ["C0", "C2"])

    assert scoped.context == "first source\n\nthird source"
    assert scoped.context_hash == digest(scoped.context)
    assert scoped.read(["C0"], radius=2)["results"][-1]["text"] == "third source"
    np.testing.assert_array_equal(scoped.matrix, index.matrix[[0, 2]])
    assert index.stats["leaves"] == 3
    assert "unselected source" in index.context


def test_stage_calls_read_committed_records_and_match_round_exactly(tmp_path):
    studio = Studio(tmp_path, "configs/repar_v5.json")
    try:
        run_id = "a" * 32
        run = tmp_path / "runs" / run_id
        job = {"id": run_id, "library_id": "b" * 32, "status": "running", "created": now()}
        write_json(run / "job.json", job)
        records = []
        for number, tag in enumerate(("v5_direct_answer", "v4_plan_1", "v4_plan_1_repair", "v4_plan_10",
                                      "v4_action_1", "v4_reflection_1", "v5_answer")):
            key = digest(tag)
            body = {"messages": [{"role": "user", "content": tag}]}
            record = {"key": key, "tag": tag, "response": {"choices": [{"message": {"content": tag}}]}}
            if number < 6:
                record["response"]["usage"] = {"prompt_tokens": 100 + number, "completion_tokens": number}
            records.append(record)
            write_json(run / "cache" / (key + ".json"), {"request": body})
            append_jsonl(run / "calls.jsonl", {"event": "completion", **record})
        append_jsonl(run / "calls.jsonl", {"event": "http_error", "error": "Mock timeout"})
        with (run / "calls.jsonl").open("ab") as stream:
            stream.write(b'{"event":"completion"')
        calls = ["runs", run_id, "calls"]
        expected_usage = {"calls": 7, "measured": 6, "prompt_tokens": 615, "completion_tokens": 15}
        assert studio.get(calls[:2], 0)["usage"] == expected_usage
        plan = studio.get([*calls, "plan", "1"], 0)["calls"]
        assert [entry["tag"] for entry in plan] == ["v4_plan_1", "v4_plan_1_repair"]
        assert plan[0]["request"]["messages"][0]["content"] == "v4_plan_1"
        assert plan[1]["response"] == records[2]["response"]
        assert studio.get([*calls, "reflection", "1"], 0)["calls"][0]["tag"] == "v4_reflection_1"
        assert not studio.get([*calls, "action", "2"], 0)["calls"]
        with pytest.raises(FileNotFoundError):
            studio.get([*calls, "plan", "0"], 0)
        write_json(run / "trace.json", {"calls": records})
        write_json(run / "job.json", {**job, "status": "ready"})
        assert studio.get([*calls, "finalizing", "0"], 0)["calls"][0]["tag"] == "v5_answer"
        assert studio.get(calls[:2], 0)["usage"] == expected_usage
        assert len(studio.get(["runs", run_id, "trace"], 0)["calls"]) == len(records)
    finally:
        studio.workers.shutdown(wait=True, cancel_futures=True)


def test_session_runs_and_unassigned_history_are_grouped(tmp_path, monkeypatch):
    studio = Studio(tmp_path, "configs/repar_v5.json")
    try:
        library_id = "a" * 32
        library = {"id": library_id, "name": "Research notes", "status": "ready", "created": now()}
        write_json(tmp_path / "libraries" / library_id / "job.json", library)
        other_library_id = "d" * 32
        write_json(tmp_path / "libraries" / other_library_id / "job.json",
                   {**library, "id": other_library_id, "name": "Project updates"})

        session = studio.submit(["sessions"], {})
        assert session["title"] == "新会话"
        assert session["auto_title"] is True
        assert "library_id" not in session

        assigned = {"id": "b" * 32, "session_id": session["id"], "library_id": library_id,
                    "status": "ready", "created": now(), "task": "Find the project date", "response": "May 2"}
        legacy = {"id": "c" * 32, "library_id": library_id, "status": "ready",
                  "created": now(), "task": "Find the archive"}
        write_json(tmp_path / "runs" / assigned["id"] / "job.json", assigned)
        write_json(tmp_path / "runs" / legacy["id"] / "job.json", legacy)

        assert [item["id"] for item in studio.session_runs(session["id"])] == [assigned["id"]]
        legacy_session_id = f"legacy-{library_id}"
        assert [item["id"] for item in studio.session_runs(legacy_session_id)] == [legacy["id"]]
        sessions = studio.get(["sessions"], 0)["sessions"]
        assert {item["id"] for item in sessions} == {session["id"], legacy_session_id}
        assert studio.get(["sessions", session["id"], "runs"], 0)["runs"][0]["id"] == assigned["id"]
        other = studio.submit(["sessions"], {"title": "Other work", "library_id": library_id})
        with pytest.raises(ValueError, match="历史问答不属于当前会话"):
            studio.submit(["runs"], {"task": "Follow up", "session_id": other["id"],
                                     "library_id": library_id, "previous_run": assigned["id"]})
        assert studio.get(["sessions", other["id"], "runs"], 0)["runs"] == []

        def query(folder, queried_library, prompt, blocks):
            assert queried_library == other_library_id
            assert "Find the project date" in prompt and "Check the updated date" in prompt
            assert "currently selected library" in prompt
            assert "May 2" not in prompt
            assert not blocks
            return {"response": "May 3"}

        monkeypatch.setattr(studio, "query", query)
        follow_up = studio.submit(["runs"], {"task": "Check the updated date", "session_id": session["id"],
                                             "library_id": other_library_id, "previous_run": assigned["id"]})
        assert follow_up["library_id"] == other_library_id
        studio.workers.shutdown(wait=True)
        assert [run["status"] for run in studio.session_runs(session["id"])] == ["ready", "ready"]
        recorded = next(entry for entry in studio.session_records() if entry["id"] == session["id"])
        assert recorded["last_library_id"] == other_library_id
        assert recorded["run_count"] == 2
    finally:
        studio.workers.shutdown(wait=True, cancel_futures=True)


def test_delete_session_removes_runs_and_keeps_other_history(tmp_path):
    studio = Studio(tmp_path, "configs/repar_v5.json")
    try:
        first, second = [studio.submit(["sessions"], {})["id"] for _ in range(2)]
        library, other_library = "a" * 32, "b" * 32
        write_json(tmp_path / "libraries" / library / "job.json",
                   {"id": library, "status": "ready", "created": now()})
        runs = [("c" * 32, first, library), ("d" * 32, first, other_library),
                ("e" * 32, second, library), ("f" * 32, None, library)]
        for run_id, session, library_id in runs:
            folder = tmp_path / "runs" / run_id
            write_json(folder / "job.json", {"id": run_id, "status": "ready", "session_id": session,
                                            "library_id": library_id, "task": "Review the notes", "created": now()})
            write_json(folder / "trace.json", {"calls": []})
        assert studio.delete_session(first) == {"id": first}
        assert not (tmp_path / "sessions" / first).exists()
        for run_id, _, _ in runs[:2]:
            assert not (tmp_path / "runs" / run_id).exists()
            with pytest.raises(FileNotFoundError):
                studio.get(["runs", run_id, "trace"], 0)
        legacy_id = f"legacy-{library}"
        studio.delete_session(legacy_id)
        assert {session["id"] for session in studio.session_records()} == {second}
        assert [run["id"] for run in studio.get(["libraries", library, "runs"], 0)["runs"]] == ["e" * 32]
        assert (tmp_path / "libraries" / library / "job.json").exists()
        busy = studio.session_runs(second)[0]
        write_json(tmp_path / "runs" / busy["id"] / "job.json", {**busy, "status": "running"})
        with pytest.raises(SessionBusyError):
            studio.delete_session(second)
        assert studio.session_runs(second)[0]["status"] == "running"
    finally:
        studio.workers.shutdown(wait=True, cancel_futures=True)
