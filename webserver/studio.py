"""Local background jobs for building and querying v5 knowledge bases."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
import re
from shutil import rmtree
from threading import RLock
import time
import traceback
from uuid import uuid4

from repar.io import append_jsonl, digest, read_json, read_jsonl, write_json

from .frozen_data import file_digest


IDENTIFIER = re.compile(r"[a-f0-9]{32}\Z")
MAX_BODY = 16 * 1024 * 1024
BLOCK_ID = re.compile(r"C(?:0|[1-9][0-9]{0,8})\Z")
ROUND_CALLS = frozenset({"plan", "action", "reflection", "control_error"})
FINAL_CALL_TAGS = {"direct_answer": "v5_direct_answer", "finalizing": "v5_answer"}


class SessionBusyError(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def run_session(run):
    return run.get("session_id") or f"legacy-{run['library_id']}"


def read_events(path, offset=0):
    rows, position = [], offset
    if path.is_file():
        with path.open("rb") as stream:
            stream.seek(offset)
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                rows.append(json.loads(line))
                position += len(line)
    return rows, position


def attach_request(folder, call):
    try:
        request = read_json(folder/"cache"/(call["key"] + ".json"))["request"]
    except FileNotFoundError:
        request = None
    return {**call, "request": request}


def recorded_calls(folder):
    trace_path = folder/"trace.json"
    if trace_path.is_file():
        return read_json(trace_path)["calls"]
    return [row for row in read_events(folder/"calls.jsonl")[0] if row["event"] == "completion"]


def summarize_usage(calls):
    prompt_tokens = completion_tokens = measured = 0
    for call in calls:
        usage = call.get("response", {}).get("usage") or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if type(prompt) is int and prompt >= 0 and type(completion) is int and completion >= 0:
            prompt_tokens += prompt
            completion_tokens += completion
            measured += 1
    return {"calls": len(calls), "measured": measured, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens}


def stage_calls(folder, stage, round_number):
    if stage in FINAL_CALL_TAGS and round_number == "0":
        pattern = re.escape(FINAL_CALL_TAGS[stage])
    elif stage in ROUND_CALLS and re.fullmatch(r"[1-9][0-9]{0,8}", round_number):
        phases = "plan|action|reflection" if stage == "control_error" else stage
        pattern = rf"v4_(?:{phases})_{round_number}(?:_repair(?:2)?)?"
    else:
        raise FileNotFoundError("Unknown trace stage")
    return [attach_request(folder, call) for call in recorded_calls(folder) if re.fullmatch(pattern, call["tag"])]


def scoped_index(index, block_ids):
    """Reuse the indexed vectors and rebuild the tree and graph over selected original blocks."""
    scoped = copy(index)
    scoped.stats = dict(index.stats)
    positions = [int(block_id[1:]) for block_id in block_ids]
    scoped.texts = [index.texts[position] for position in positions]
    scoped.context = "\n\n".join(scoped.texts)
    scoped.context_hash = digest(scoped.context)
    scoped.spans = []
    start = 0
    for text in scoped.texts:
        end = start + len(text)
        scoped.spans.append((start, end))
        start = end + 2
    scoped.matrix = index.matrix[positions]
    scoped.chunk_tokens = [index.chunk_tokens[position] for position in positions]
    scoped.knowledge = [index.knowledge[position] for position in positions]
    scoped.build_tree(index.config["tree_fanout"])
    scoped.stats.update(context_sha256=scoped.context_hash, characters=len(scoped.context),
                        leaves=len(scoped.spans), tree_nodes=len(scoped.nodes),
                        observed_max_chunk_tokens=max(scoped.chunk_tokens), build_seconds=0)
    scoped._build_graph()
    return scoped


class Studio:
    def __init__(self, root, config_path, *, examples=None):
        self.root = Path(root)
        self.config_path = Path(config_path)
        self.examples = examples or {}
        self.metadata_lock = RLock()
        self.example_lock = RLock()
        self.workers = ThreadPoolExecutor(max_workers=1, thread_name_prefix="repar-studio")
        for kind in ("libraries", "runs"):
            for path in (self.root/kind).glob("*/job.json"):
                job = read_json(path)
                if job["status"] in {"queued", "running"}:
                    write_json(path, {**job, "status": "error", "error": "服务重启中断了任务，请重新提交。"})

    def job_path(self, kind, job_id):
        if not IDENTIFIER.fullmatch(job_id):
            raise FileNotFoundError(job_id)
        return self.root/kind/job_id

    def records(self, kind):
        with self.metadata_lock:
            return sorted((read_json(p) for p in (self.root/kind).glob("*/job.json")),
                          key=lambda row: row["created"], reverse=True)

    def library_records(self):
        return self.records("libraries") + [example[0] for example in self.examples.values()]

    def library_metadata(self, library_id):
        if library_id in self.examples:
            return self.examples[library_id][0]
        return self.metadata(self.job_path("libraries", library_id))

    def session_records(self):
        with self.metadata_lock:
            sessions = [row for row in self.records("sessions") if not self.session_deleted(row["id"])]
            libraries = {item["id"]: item for item in self.library_records()}
            grouped = {}
            for run in self.records("runs"):
                session_id = run_session(run)
                if not self.session_deleted(session_id):
                    grouped.setdefault(session_id, []).append(run)
        for session_id, runs in grouped.items():
            if session_id.startswith("legacy-") and runs[0]["library_id"] in libraries:
                sessions.append({"id": session_id, "title": "历史对话", "library_id": runs[0]["library_id"],
                                 "legacy": True, "created": runs[-1]["created"]})
        items = []
        for session in sessions:
            runs = grouped.get(session["id"], [])
            latest_library_id = runs[0]["library_id"] if runs else session.get("library_id")
            library = libraries.get(latest_library_id, {})
            title = session["title"]
            if session.get("auto_title") and runs:
                title = " ".join(runs[-1]["task"].split())[:48]
            items.append({**session, "title": title, "last_library_id": latest_library_id,
                          "library_name": library.get("name", ""), "run_count": len(runs),
                          "updated": runs[0]["created"] if runs else session["created"]})
        return sorted(items, key=lambda row: row["updated"], reverse=True)

    def session_runs(self, session_id):
        with self.metadata_lock:
            if self.session_deleted(session_id):
                raise FileNotFoundError(session_id)
            return [run for run in self.records("runs") if run_session(run) == session_id]

    def deletion_marker(self, session_id):
        if not IDENTIFIER.fullmatch(session_id.removeprefix("legacy-")):
            raise FileNotFoundError(session_id)
        return self.root/"deleted_sessions"/(session_id + ".json")

    def session_deleted(self, session_id):
        return self.deletion_marker(session_id).is_file()

    def delete_session(self, session_id):
        marker = self.deletion_marker(session_id)
        legacy = session_id.startswith("legacy-")
        with self.metadata_lock:
            runs = [row for row in self.records("runs") if run_session(row) == session_id]
            run_ids = read_json(marker)["runs"] if marker.is_file() else [row["id"] for row in runs]
            paths = [self.job_path("runs", run_id) for run_id in run_ids]
            if not legacy:
                paths.append(self.job_path("sessions", session_id))
            if not marker.exists() and not any((path/"job.json").is_file() for path in paths):
                raise FileNotFoundError(session_id)
            if any(row["status"] in {"queued", "running"} for row in runs):
                raise SessionBusyError("会话中有排队或运行中的问答，请在结束后删除。")
            root = self.root.resolve()
            if any(path.resolve() != root/path.parent.name/path.name for path in paths):
                raise ValueError("会话目录超出工作台的数据范围。")
            write_json(marker, {"runs": run_ids})
            for path in paths:
                if path.exists():
                    rmtree(path)
            marker.unlink()
        return {"id": session_id}

    def metadata(self, folder):
        with self.metadata_lock:
            return read_json(folder/"job.json")

    def save_metadata(self, folder, job):
        with self.metadata_lock:
            write_json(folder/"job.json", job)

    def get(self, parts, offset):
        if parts == ["libraries"]:
            return {"libraries": self.library_records()}
        if parts == ["sessions"]:
            return {"sessions": self.session_records()}
        if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "runs":
            session_id = parts[1]
            runs = self.session_runs(session_id)
            if not runs and not (self.job_path("sessions", session_id) / "job.json").is_file():
                raise FileNotFoundError(parts[1])
            return {"runs": runs}
        if len(parts) < 2 or parts[0] not in {"libraries", "runs"}:
            raise FileNotFoundError("Unknown studio resource")
        kind, job_id, *tail = parts
        folder = self.job_path(kind, job_id)
        with self.metadata_lock:
            job = self.library_metadata(job_id) if kind == "libraries" else self.metadata(folder)
            if kind == "runs" and self.session_deleted(run_session(job)):
                raise FileNotFoundError(job_id)
        if not tail:
            rows, position = read_events(folder/"events.jsonl", offset)
            result = {"job": job, "events": rows, "offset": position}
            if kind == "runs":
                result["usage"] = summarize_usage(recorded_calls(folder))
            return result
        if kind == "libraries" and tail == ["runs"]:
            with self.metadata_lock:
                return {"runs": [run for run in self.records("runs")
                                 if run["library_id"] == job_id and not self.session_deleted(run_session(run))]}
        if kind == "libraries" and tail == ["index"] and job["status"] == "ready":
            if job_id in self.examples:
                return self.example_index_data(job_id)
            return read_json(folder/"index.json")
        if kind == "runs" and len(tail) == 3 and tail[0] == "calls":
            return {"calls": stage_calls(folder, tail[1], tail[2])}
        if kind == "runs" and tail == ["trace"] and job["status"] in {"ready", "error"}:
            trace = folder/"trace.json"
            if trace.is_file():
                result = read_json(trace)
            else:
                events = self.get(parts[:2], 0)["events"]
                result = {"trace": events, "calls": [], "error": job["error"], "state": {}, "termination": "error"}
            result["calls"] = [attach_request(folder, call) for call in result["calls"]]
            return result
        raise FileNotFoundError("Unknown studio resource")

    def submit(self, parts, data):
        if not isinstance(data, dict):
            raise ValueError("请求须为 JSON 对象。")
        if parts == ["libraries"]:
            name, documents = data.get("name"), data.get("documents")
            if not isinstance(name, str) or not name.strip() or len(name) > 120:
                raise ValueError("请输入知识库名称，最多 120 个字符。")
            if not isinstance(documents, list) or not documents:
                raise ValueError("请上传至少一个 UTF-8 文本文件。")
            texts, sources, start = [], [], 0
            for doc in documents:
                if not isinstance(doc, dict):
                    raise ValueError("无效的文档对象。")
                filename, content = doc.get("name"), doc.get("text")
                if not isinstance(filename, str) or not filename.strip() or len(filename) > 512:
                    raise ValueError("无效的文件名。")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"文件正文为空。{filename}")
                texts.append(content)
                end = start + len(content)
                sources.append({"name": filename, "start": start, "end": end})
                start = end + 2
            return self.enqueue("libraries", {"name": name.strip(), "sources": sources}, self.build, "\n\n".join(texts))
        if parts == ["sessions"]:
            title = data.get("title", "")
            library_id = data.get("library_id")
            if not isinstance(title, str) or len(title.strip()) > 120:
                raise ValueError("会话名称最多 120 个字符。")
            if "documents" in data:
                if library_id is not None:
                    raise ValueError("请选择已有知识库或上传新资料。")
                library = self.submit(["libraries"], {"name": data.get("library_name"), "documents": data["documents"]})
                library_id = library["id"]
            elif isinstance(library_id, str):
                library = self.library_metadata(library_id)
                if library["status"] != "ready":
                    raise ValueError("请选择已就绪的知识库。")
            elif library_id is not None:
                raise ValueError("无效的知识库 ID。")
            else:
                library = None
            session = {"id": uuid4().hex, "title": title.strip() or "新会话",
                       "auto_title": not title.strip(), "created": now()}
            if library is not None:
                session["library_id"] = library_id
            self.save_metadata(self.job_path("sessions", session["id"]), session)
            return {**session, "run_count": 0, **({"library": library,
                    "library_name": library["name"], "status": library["status"]} if library else {})}
        if parts == ["runs"]:
            library_id, task = data.get("library_id"), data.get("task")
            if not isinstance(library_id, str) or not isinstance(task, str) or not task.strip() or len(task) > 20000:
                raise ValueError("请选择知识库并输入 1 至 20,000 字符的问题。")
            library = self.library_metadata(library_id)
            if library["status"] != "ready":
                raise ValueError("请等待知识库完成索引。")
            blocks = data.get("block_ids", [])
            if (not isinstance(blocks, list) or len(blocks) > 24 or
                    any(not isinstance(block, str) or not BLOCK_ID.fullmatch(block) for block in blocks)):
                raise ValueError("请选择最多 24 个有效的原文块。")
            if blocks and any(int(block[1:]) >= library["stats"]["leaves"] for block in blocks):
                raise ValueError("原文块不属于当前知识库。")
            blocks = sorted(set(blocks), key=lambda block: int(block[1:]))
            session_id = data.get("session_id")
            legacy_id = f"legacy-{library_id}"
            if session_id is not None and not isinstance(session_id, str):
                raise ValueError("无效的会话 ID。")
            previous = data.get("previous_run")
            prompt = task.strip()
            if previous:
                if not isinstance(previous, str):
                    raise ValueError("无效的历史运行 ID。")
                prior = self.metadata(self.job_path("runs", previous))
                if prior["status"] != "ready":
                    raise ValueError("请选择已完成的历史问答。")
                if run_session(prior) != (session_id or legacy_id):
                    raise ValueError("历史问答不属于当前会话。")
                history = f"Previous question:\n{prior['task']}"
                if prior["library_id"] == library_id:
                    history += f"\n\nPrevious answer:\n{prior['response']}"
                prompt = ("Use the conversation context only to resolve references. "
                          "Retrieve evidence for the new answer from the currently selected library.\n\n"
                          f"{history}\n\nNew question:\n{prompt}")
            with self.metadata_lock:
                target = session_id if session_id is not None else legacy_id
                if self.session_deleted(target):
                    raise FileNotFoundError(target)
                if session_id is not None:
                    if session_id.startswith("legacy-"):
                        if not self.session_runs(session_id):
                            raise FileNotFoundError(session_id)
                    else:
                        self.metadata(self.job_path("sessions", session_id))
                if previous:
                    self.metadata(self.job_path("runs", previous))
                return self.enqueue("runs", {"library_id": library_id, "session_id": session_id,
                                             "task": task.strip(), "previous_run": previous, "block_ids": blocks},
                                    self.query, library_id, prompt, blocks)
        raise FileNotFoundError("Unknown studio resource")

    def enqueue(self, kind, metadata, worker, *arguments):
        job = {"id": uuid4().hex, "status": "queued", "created": now(), **metadata}
        folder = self.job_path(kind, job["id"])
        self.save_metadata(folder, job)
        self.workers.submit(self.run, folder, job, worker, *arguments)
        return job

    def run(self, folder, job, worker, *arguments):
        self.save_metadata(folder, {**job, "status": "running"})
        started = time.perf_counter()
        try:
            job = {**job, **worker(folder, *arguments), "status": "ready"}
        except Exception as error:
            (folder/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
            job = {**job, "status": "error", "error": str(error)}
            if "library_id" in job and not (folder/"trace.json").exists():
                write_json(folder/"trace.json", {"trace": [], "calls": [], "error": str(error), "state": {},
                                                "termination": "error"})
        self.save_metadata(folder, {**job, "seconds": time.perf_counter()-started, "finished": now()})

    def build(self, folder, context):
        from repar.dense_index import DenseKnowledgeIndex, V5
        from .knowledge_data import tree_graph_data

        config = read_json(self.config_path)
        if config.get("agent_version") != V5:
            raise ValueError("工作台需要 react-v5 配置。")
        (folder/"source.txt").write_bytes(context.encode("utf-8"))
        write_json(folder/"config.json", config)
        events = folder/"events.jsonl"
        append_jsonl(events, {"stage": "embedding", "characters": len(context)})
        index = DenseKnowledgeIndex(context, config, build_graph=False)
        append_jsonl(events, {"stage": "graph", "leaves": index.stats["leaves"], "tree_nodes": len(index.nodes)})
        index._build_graph()
        append_jsonl(events, {"stage": "saving", "entities": len(index.entity_graph.entities)})
        index.save(folder/"index.pkl")
        write_json(folder/"index.json", tree_graph_data(index, context))
        return {"stats": index.stats, "model": config["model"]}

    def load_example_index(self, library_id):
        _, path, checksum, context_sha = self.examples[library_id]
        if file_digest(path) != checksum:
            raise ValueError("示例知识库的冻结索引校验失败。")
        with path.open("rb") as stream:
            index = pickle.load(stream)
        if index.context_hash != context_sha or digest(index.context) != context_sha:
            raise ValueError("示例知识库的原文校验失败。")
        return index

    def example_index_data(self, library_id):
        from .knowledge_data import tree_graph_data

        path = self.job_path("libraries", library_id) / "index.json"
        with self.example_lock:
            if not path.is_file():
                index = self.load_example_index(library_id)
                write_json(path, tree_graph_data(index, index.context))
        return read_json(path)

    def query(self, folder, library_id, prompt, block_ids):
        from repar.model_support import BoundedLLMClient
        from repar.v5 import answer

        if library_id in self.examples:
            config = read_json(self.config_path)
            index = self.load_example_index(library_id)
            embedding = config["embedding"]
            frozen_embedding = index.config["embedding"]
            if embedding["manifest_sha256"] == frozen_embedding["manifest_sha256"]:
                index.config = config
            else:
                index.config = {**config, "embedding": {**frozen_embedding, "query_device": embedding["query_device"]}}
        else:
            library = self.job_path("libraries", library_id)
            config = read_json(library/"config.json")
            with (library/"index.pkl").open("rb") as stream:
                index = pickle.load(stream)
        if block_ids:
            index = scoped_index(index, block_ids)
            append_jsonl(folder/"events.jsonl", {"stage": "source_scope", "block_ids": block_ids})
            prompt = ("The user restricted source verification to selected passages of the knowledge base. "
                      "If those passages do not support an answer, say so.\n\n" + prompt)
        client = BoundedLLMClient(config, folder/"cache", call_seconds=1800, journal=folder/"calls.jsonl")
        try:
            result = answer(prompt, index, client, config, event_path=folder/"events.jsonl")
        except Exception as error:
            write_json(folder/"trace.json", {"trace": read_jsonl(folder/"events.jsonl"), "calls": client.calls,
                                            "error": str(error), "state": {}, "termination": "error"})
            raise
        if block_ids:
            result["source_map"] = {f"C{i}": block for i, block in enumerate(block_ids)}
        write_json(folder/"trace.json", result)
        return {"response": result["response"], "termination": result["termination"],
                "evidence_ids": result["evidence_ids"], "model": config["model"]}
