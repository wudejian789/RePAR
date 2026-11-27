"""Full evaluation adapters for v3; the agent has no benchmark dependency."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback

import requests

from repar.benchmarks.math_exam import public_question, task_text, solver_messages, score_choice, aggregate_grades, stitch_context
from repar.benchmarks.math_judge import grade_once
from repar.evaluation import qa_text, official_prompt, extract_answer
from repar.io import read_json, read_jsonl, write_json, append_jsonl, digest, load_public_samples, public_item
from repar.model_support import BoundedLLMClient
from repar.v3 import research


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now().astimezone().isoformat()


def unique(path, field="id", expected=None):
    values = read_jsonl(path)
    result = {r[field]: r for r in values}
    if len(result) != len(values): raise ValueError(f"Duplicate IDs in {path}")
    if expected is not None and set(result) != set(expected): raise ValueError(f"Incomplete coverage: {path}")
    return result


def freeze(config_path):
    config = read_json(config_path)
    expected_versions = {"retrieval_core_r1": "react-v3-core-r1", "backbone_v2": "react-v2",
                         "baseline_direct": "direct", "baseline_bm25": "bm25-rag-v1",
                         "bound_views_v4": "react-v4", "bound_views_v4_r2": "react-v4-r2",
                         "bound_views_v4_r3": "react-v4-r3", "bound_views_v4_r4": "react-v4-r4",
                         "bound_views_v4_r5": "react-v4-r5", "bound_views_v4_r6": "react-v4-r6", "bound_views_v4_r7": "react-v4-r7", "bound_views_v4_r8": "react-v4-r8", "bound_views_v4_r9": "react-v4-r9", "bound_views_v4_r10": "react-v4-r10",
                         "bound_views_v4_r11": "react-v4-r11", "bound_views_v4_r12": "react-v4-r12",
                         "bound_views_v4_r13": "react-v4-r13", "bound_views_v4_r14": "react-v4-r14",
                         "bound_views_v4_r15": "react-v4-r15", "bound_views_v4_r16": "react-v4-r16",
                         "bound_views_v4_r17": "react-v4-r17", "bound_views_v4_r18": "react-v4-r18",
                         "bound_views_v4_r19": "react-v4-r19", "bound_views_v4_r20": "react-v4-r20", "bound_views_v4_r22": "react-v4-r22", "bound_views_v4_r23": "react-v4-r23", "bound_views_v4_r24": "react-v4-r24", "bound_views_v4_r25": "react-v4-r25", "bound_views_v4_r26": "react-v4-r26", "bound_views_v4_r27": "react-v4-r27",
                         "bound_views_v4_r27_hope": "react-v4-r27-hope",
                         "bound_views_v4_r27_hope2": "react-v4-r27-hope2",
                         "bound_views_v4_r27_hope3": "react-v4-r27-hope3",
                         "bound_views_v4_r60": "react-v4-r60",
                         "bound_views_v4_r61": "react-v4-r61",
                         "prior_answer_v5": "react-v5"}
    if config.get("variant") not in expected_versions:
        raise ValueError("Historical conditions require their frozen runner; new runs need a supported evaluation protocol")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    solver = read_json(config["solver_config"])
    if solver.get("agent_version") != expected_versions[config["variant"]]:
        raise ValueError("Evaluation protocol and workflow version do not match")
    files = sorted({*Path("src/repar").rglob("*.py"), *Path("baselines/BM25_RAG").rglob("*.py"),
                    Path("baselines/__init__.py"), Path("scripts/run_repar_v3.py"), Path("scripts/research_kb_v3.py"),
                    Path("tests/test_v3.py"), Path(config_path), Path(config["solver_config"]),
                    Path("pyproject.toml"), Path("third_party/prompts/0shot_rag.txt"),
                    *[Path(p) for p in config.get("additional_snapshot_files", [])]})
    identity = {"config": config, "solver": solver, "files": {p.as_posix(): sha(p) for p in files},
                "data": {config[k]: sha(config[k]) for k in ("math_questions", "math_textbook", "longbench_data")}}
    if config.get("immutable_references"):
        references = sorted({p for pattern in config["immutable_references"] for p in Path(".").glob(pattern) if p.is_file()})
        if not references:
            raise ValueError("Immutable reference patterns did not match")
        identity["references"] = {p.as_posix(): sha(p) for p in references}
    path = output/"protocol.json"
    if path.exists():
        if read_json(path)["identity"] != identity: raise ValueError("Frozen experiment changed; create a new version")
        return config, output
    deployments = {}
    for role, c in (("solver", solver), ("judge", config["judge"])):
        if role == 'judge' and config.get('judge_deployment_reference'):
            reference = config['judge_deployment_reference']
            if reference not in identity.get('references', {}):
                raise ValueError('The judge deployment reference must be frozen with this experiment')
            prior = read_json(reference)
            if prior['identity']['config']['judge'] != c:
                raise ValueError('Recorded judge configuration differs')
            deployments[role] = {**prior['deployments']['judge'], 'observed_in': reference}
            continue
        models = requests.get(c["base_url"]+"/v1/models", timeout=30).json()
        if c["model"] not in [m["id"] for m in models["data"]]: raise ValueError("Unexpected model service")
        deployments[role] = {"models": models, "props": requests.get(c["base_url"]+"/props", timeout=30).json()}
    for source in files:
        target = output/"source"/source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    write_json(path, {"frozen_at": timestamp(), "identity": identity, "deployments": deployments,
                      "limits": "No remote judge weight hash independently measured; shared hardware, no isolated speed claim"})
    print("FROZEN", config["experiment_id"], flush=True)
    return config, output


class Suite:
    def __init__(self, config_path):
        self.config = read_json(config_path)
        self.output = Path(self.config["output"])
        self.solver = read_json(self.config["solver_config"])
        self.protocol = read_json(self.output/"protocol.json")
        for name, expected in self.protocol["identity"]["files"].items():
            if sha(name) != expected: raise ValueError(f"Runtime differs from snapshot: {name}")
        for name, expected in self.protocol["identity"]["data"].items():
            if sha(name) != expected: raise ValueError(f"Data changed: {name}")

    def client(self, role="solver", seed=None):
        config = dict(self.solver if role == "solver" else self.config["judge"])
        if seed is not None: config["seed"] = seed
        return BoundedLLMClient(config, self.output/"cache"/role, call_seconds=1800,
                                journal=self.output/f"{role}_calls.jsonl")

    def load_index(self, context, kind):
        relevant = {k: self.solver.get(k) for k in ("chunk_words", "tree_fanout", "graph_model", "index_version")}
        key = digest(context+json.dumps(relevant, sort_keys=True))
        directory = Path(self.config["math_parent"])/"indexes" if kind == "math" else Path("artifacts/indexes_v2")
        path = directory/(key+".pkl")
        if not path.exists(): raise ValueError(f"Expected unchanged historical index missing: {path}")
        with path.open("rb") as stream: index = pickle.load(stream)
        if index.context != context: raise ValueError("Historical index source mismatch")
        return index, {"path": path.as_posix(), "context_sha256": digest(context), "reused": True,
                       "stats": index.stats, "note": "No source cleaning/reformatting, extraction, or repartitioning"}

    def generate_math(self):
        questions = read_jsonl(self.config["math_questions"])
        if len(questions) != 19: raise ValueError("Expected all19 math questions")
        context, meta = stitch_context(self.config["math_textbook"])
        index, index_info = self.load_index(context, "math")
        write_json(self.output/"math/index_reuse.json", index_info)
        path = self.output/"math/predictions.jsonl"
        done = unique(path)
        for item in questions:
            if item["id"] in done: continue
            public = public_question(item)
            row = self.generate(item["id"], task_text(item), index, "math")
            row["question"] = public
            append_jsonl(path, row)
            print("DONE MATH", item["id"], row["status"], f"{row['seconds']:.1f}s", flush=True)
        unique(path, expected=[q["id"] for q in questions])
        self.lock_predictions("math")

    def generate_longbench(self):
        samples = load_public_samples(self.config["longbench_data"])
        if len(samples) != 125: raise ValueError("Expected all125 Multi-Document QA")
        path = self.output/"longbench/predictions.jsonl"
        done = unique(path)
        for i, labelled in enumerate(samples):
            if labelled["_id"] in done: continue
            item = public_item(labelled)
            print("START LONGBENCH", i+1, "/125", item["_id"], flush=True)
            started = time.perf_counter()
            index, reuse = self.load_index(item["context"], "longbench")
            row = self.generate(item["_id"], qa_text(item), index, "longbench", longbench_item=item)
            row["index_load_wall_seconds"] = time.perf_counter()-started-row["seconds"]
            row["index_reuse"] = reuse
            # Gold is used only after the final response; never enters agent/index.
            row.update({"pred": extract_answer(row["response"]), "gold": labelled["answer"]})
            row["correct"] = row["pred"] == row["gold"]
            append_jsonl(path, row)
            print("DONE LONGBENCH", i+1, "/125", row["status"], f"{row['seconds']:.1f}s", flush=True)
        unique(path, expected=[q["_id"] for q in samples])
        self.lock_predictions("longbench")

    def generate(self, qid, task, index, dataset, longbench_item=None):
        client = self.client()
        started = time.perf_counter()
        row = {"id": qid, "time": timestamp(), "status": "ok", "task": task, "response": "",
               "context_sha256": index.context_hash, "agent_version": self.solver["agent_version"], "trace": []}
        try:
            result = self.research_result(task, index, client, dataset, qid, longbench_item=longbench_item)
            row.update({"trace": result.trace, "evidence_ids": result.evidence_ids, "evidence_records": result.evidence_records,
                        "state": result.state, "termination": result.termination})
            messages = self.answer_messages(task, result, longbench_item)
            if longbench_item is None:
                maximum = self.solver["answer_max_tokens"]
                tag = "math_repar_answer"
            else:
                maximum = self.config["longbench_final_max_tokens"]
                tag = "v2_repar_answer"
                client.config["answer_grammar"] = self.config["longbench_answer_grammar"]
            rendered = client.post("/apply-template", {"messages": messages, "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
            row["rendered_input_tokens"] = len(client.tokenize(rendered))
            if row["rendered_input_tokens"]+maximum+128 > self.solver["context_window"]:
                raise ValueError("Final input exceeds configured model window")
            row["response"] = client.complete(messages, maximum, tag=tag)
            if not row["response"].strip():
                raise ValueError("Empty final response")
        except Exception as exc:
            row.update({"status": "error", "error": str(exc), "traceback": traceback.format_exc()})
        row.update({"seconds": time.perf_counter()-started, "calls": client.calls})
        write_json(self.output/dataset/"traces"/(qid+".json"), row)
        return row

    def answer_messages(self, task, result, longbench_item):
        if longbench_item is None:
            return solver_messages(task, result.evidence or "未取得相关教材内容。")
        context = result.evidence or "No source evidence was obtained."
        return [{"role": "user", "content": official_prompt(longbench_item, context, rag=True)}]

    def research_result(self, task, index, client, dataset, qid, *, longbench_item=None):
        return research(task, index, client, self.solver,
                        event_path=self.output/dataset/"events"/(qid+".jsonl"))

    def lock_predictions(self, dataset):
        path = self.output/dataset/"generation_manifest.json"
        value = {"predictions_sha256": sha(self.output/dataset/"predictions.jsonl"),
                 "count": 19 if dataset == "math" else 125, "note": "All predictions fixed before summary/judge scoring"}
        if path.exists() and read_json(path) != value: raise ValueError("Locked predictions changed")
        if not path.exists(): write_json(path, value)

    def grade_math(self):
        questions = {q["id"]: q for q in read_jsonl(self.config["math_questions"])}
        predictions = unique(self.output/"math/predictions.jsonl", expected=questions)
        self.lock_predictions("math")
        path = self.output/"math/grades.jsonl"
        previous = unique(path)
        clients = [self.client("judge", seed) for seed in self.config["judge"]["replicate_seeds"]]
        for qid, item in questions.items():
            if qid in previous: continue
            p = predictions[qid]
            row = {"id": qid, "response_sha256": digest(p["response"]), "grades": [], "judge_calls": []}
            if item["question_type"] in {"single_choice", "multiple_choice"}: row.update(score_choice(item, p["response"]))
            elif p["status"] != "ok": row.update({"score": 0, "fully_correct": False, "strict_score": 0, "max_score": item["score"]})
            else:
                for replicate, client in enumerate(clients):
                    first = len(client.calls)
                    grade = grade_once(item, p["response"], client, self.config["judge"]["max_tokens"], f"v3_judge_{qid}_r{replicate+1}")
                    row["grades"].append(grade); row["judge_calls"].extend(client.calls[first:])
                    write_json(self.output/"math/judgements"/(qid+".json"), row)
                    if replicate == 1:
                        a, b = row["grades"]
                        if {u["id"]: u["credit"] for u in a["units"]} == {u["id"]: u["credit"] for u in b["units"]}: break
                row.update(aggregate_grades(item, row["grades"]))
            append_jsonl(path, row)
            print("GRADED MATH", qid, "replicates", len(row["grades"]), flush=True)
        unique(path, expected=questions)

    def report(self):
        math_ids = [q["id"] for q in read_jsonl(self.config["math_questions"])]
        lb_ids = [q["_id"] for q in load_public_samples(self.config["longbench_data"])]
        math = unique(self.output/"math/predictions.jsonl", expected=math_ids)
        grades = unique(self.output/"math/grades.jsonl", expected=math_ids)
        lb = unique(self.output/"longbench/predictions.jsonl", expected=lb_ids)
        self.lock_predictions("math"); self.lock_predictions("longbench")
        for qid, g in grades.items():
            if g["response_sha256"] != digest(math[qid]["response"]): raise ValueError("Grade response mismatch")
        previous_lb = {r["_id"]: r for r in read_jsonl(self.config["longbench_parent"])}
        summary = {"completed_at": timestamp(), "math": {"score": sum(g["score"] for g in grades.values()),
             "fully_correct": sum(g["fully_correct"] for g in grades.values()), "count": len(math),
             "errors": sum(p["status"] != "ok" for p in math.values())},
             "longbench": {"correct": sum(p["correct"] for p in lb.values()), "count": len(lb),
             "accuracy": sum(p["correct"] for p in lb.values())/len(lb), "errors": sum(p["status"] != "ok" for p in lb.values()),
             "old_correct": sum(r["judge"] for r in previous_lb.values()),
             "new_only_correct": [qid for qid, p in lb.items() if p["correct"] and not previous_lb[qid]["judge"]],
             "old_only_correct": [qid for qid, p in lb.items() if previous_lb[qid]["judge"] and not p["correct"]]},
             "source_policy": self.config["source_policy"], "exposure": self.config["exposure"]}
        for name, predictions in (("math", math), ("longbench", lb)):
            calls = [c for r in predictions.values() for c in r["calls"]]
            summary[name].update({"calls": len(calls), "cache_hits": sum(c["cached"] for c in calls),
                "prompt_tokens": sum(c["response"].get("usage", {}).get("prompt_tokens", 0) for c in calls),
                "completion_tokens": sum(c["response"].get("usage", {}).get("completion_tokens", 0) for c in calls),
                "repairs": sum("_repair" in c["tag"] for c in calls),
                "tool_calls": sum(len(t.get("executed", [])) for p in predictions.values() for t in p["trace"])})
        write_json(self.output/"summary.json", summary)
        display = self.config.get("method_display", "v3检索核心修订")
        lines = [f"# RePAR {display}完整评测", "", f"作答底座为{self.solver['model']}，关闭thinking；教材/原文和v2索引保持原样。", "",
                 "| 条件 | MATH实验折算分/150 | MATH整题全对 | LongBench Multi-Document QA |", "|---|---:|---:|---:|",
                 "| 历史直接回答 | 18 | 3/19 | 20.8%（26/125） |", "| 历史BM25 | 25 | 5/19 | 23.2%（29/125） |",
                 "| 历史v2 | 14.5 | 2/19 | 27.2%（34/125） |",
                 f"| {display}本次 | {summary['math']['score']:.3f} | {summary['math']['fully_correct']}/19 | {100*summary['longbench']['accuracy']:.1f}%（{summary['longbench']['correct']}/125） |", "",
                 "历史条件是引用原结果，并非本次重跑。数学沿用v2的solver_messages和4096-token原始回答；LongBench沿用官方RAG提示及128-token约束。"
                 "本条件不增加解题工具、最终答案JSON、字段恢复、答案复核或来源分类作答门槛。数学分为原实验rubric折算分。", "",
                 "两套测试及历史结果此前已被查看，此次为开发实验，不是未见测试确认。未按题号加规则、未修改数据、未挑选问题或在题级切换方法。"
                 "单次结果不能保证任意数据集提升；若不满足目标，也完整报告负结果。", "", "## 原始汇总", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""]
        if self.config.get("report_filename"):
            lines[2:2] = [f"生成时间：{timestamp()}（本地Asia/Shanghai）", ""]
        (self.output/self.config.get("report_filename", "REPORT.md")).write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_experiment_core.json")
    parser.add_argument("--phase", choices=("math", "longbench", "report", "all"), default="all")
    args = parser.parse_args()
    variant = read_json(args.config).get("variant")
    if variant in {"bound_views_v4", "bound_views_v4_r2", "bound_views_v4_r3", "bound_views_v4_r4", "bound_views_v4_r5", "bound_views_v4_r6", "bound_views_v4_r7", "bound_views_v4_r8", "bound_views_v4_r9", "bound_views_v4_r10", "bound_views_v4_r11", "bound_views_v4_r12", "bound_views_v4_r13", "bound_views_v4_r14", "bound_views_v4_r15", "bound_views_v4_r16", "bound_views_v4_r17", "bound_views_v4_r18", "bound_views_v4_r19", "bound_views_v4_r20", "bound_views_v4_r22", "bound_views_v4_r23", "bound_views_v4_r24", "bound_views_v4_r25", "bound_views_v4_r26", "bound_views_v4_r27", "bound_views_v4_r27_hope", "bound_views_v4_r27_hope2", "bound_views_v4_r27_hope3", "bound_views_v4_r60", "bound_views_v4_r61"}:
        from repar.benchmarks.v4_suite import V4Suite
        suite = V4Suite(args.config)
    elif variant in {"baseline_direct", "baseline_bm25"}:
        from repar.benchmarks.baseline_suite import BaselineSuite
        suite = BaselineSuite(args.config)
    elif variant == "backbone_v2":
        from repar.benchmarks.backbone_suite import V2Suite
        suite = V2Suite(args.config)
    else:
        suite = Suite(args.config)
    if args.phase in {"math", "all"}: suite.generate_math(); suite.grade_math()
    if args.phase in {"longbench", "all"}: suite.generate_longbench()
    if args.phase in {"report", "all"}: suite.report()
