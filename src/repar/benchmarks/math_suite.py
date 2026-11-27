"""Reproducible three-method exam experiment; no changes to core algorithms."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests

from baselines.BM25_RAG import BM25Retriever, retrieve_context
from repar.benchmarks.math_exam import (aggregate_grades, score_choice, solver_messages,
                                       stitch_context, task_text, unit_ids)
from repar.benchmarks.math_judge import calibration_cases, grade_once
from repar.index_v2 import KnowledgeIndex
from repar.io import append_jsonl, digest, read_json, read_jsonl, write_json
from repar.model_support import BoundedLLMClient
from repar.workflow import research


RULES = {
    "version": "math-exam-rubric-v4-verified-verbal-rating",
    "selection": "all 19 unique questions, all three methods; no score-based selection or tuning",
    "direct": "full context attempted; fixed head/tail context token truncation, question preserved; same subset for all questions",
    "solver": "existing Qwen3.5-0.8B standard Q4, no thinking/MTP; same answer prompt and 4096-token budget for all methods",
    "retrieval": "existing chunk_words=240/top_k=24 BM25 and existing 4-step general RePAR v2 controller; complete stitched context",
    "choices": "single exact; multiple exact for full correctness, proportional correct-subset points; any wrong selection=0",
    "fill": "judge mathematical equivalence, binary credit, complete root/value set required",
    "constructed": "equal weight per dataset subquestion; rubric credit in {0,.25,.5,.75,1}; no claimed official subquestion marks",
    "replicates": "verify key candidate assertions before credit; two independent anonymous judgements with distinct fixed seeds; any unit credit disagreement triggers third; per-unit median",
    "calibration_history": "Before any exam generation, v1-v3 gave inconsistent numerical partial credits to invalid or repeated synthetic proofs. v4 uses verified verbal ratings mapped deterministically to the same rubric credits. All rejected calibration protocols remain in artifacts/math_2025_national_i_v1 through v3. A separate numeric-schema probe passed; no proven service enum bug.",
    "execution_history": "v4 calibration passed; direct blocked before HTTP by incorrect JSON-serialization token counting. v5 counts the server-rendered chat template. Six identical BM25 completions and five calibration calls may reuse v4 request-exact cache; no exam grades were inspected and no answers regenerated to improve scores.",
    "metrics": "experimental score /150, strict whole-question accuracy and full-credit-only weighted score; multi-choice proxy separately visible",
    "failures": "retain errors and assign zero for solver failure; unresolved judge failure blocks report; no answer-generation retries based on scores",
    "exposure": "dataset/reference audit predates this run; references read for adapter design, not an unseen-test confirmation; no solver tuning on these answers",
    "limitations": "English-only entity extractor on Chinese OCR text; unfiltered source noise; LLM judging is not official/human marking",
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now().astimezone().isoformat()


def unique_rows(path, expected=None):
    rows = read_jsonl(path)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs: {path}")
    if expected is not None and set(ids) != set(expected):
        raise ValueError(f"Incomplete ID coverage: {path}")
    return {r["id"]: r for r in rows}


class Suite:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = read_json(config_path)
        self.output = Path(self.config["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.solver = read_json(self.config["solver_config"])
        self.solver.update(self.config["solver_overrides"])
        self.solver.pop("answer_grammar", None)
        self.solver.pop("controller_schema", None)
        self.questions = list(unique_rows(self.config["questions"]).values())
        if len(self.questions) != 19 or sum(q["score"] for q in self.questions) != 150:
            raise ValueError("Expected frozen national-I dataset: 19 questions/150 points")
        self.rules = RULES
        if "raptor" in self.config["methods"]:
            self.rules = {**RULES,
                "selection": "All 19 unique questions for each configured method.",
                "retrieval": RULES["retrieval"] + "; RAPTOR uses one shared textbook tree",
                "execution_history": "Independent RAPTOR run with its own request cache."}
        self.by_id = {q["id"]: q for q in self.questions}
        self.context, self.context_manifest = stitch_context(self.config["textbook"])

    def client(self, role, seed=None):
        config = dict(self.solver if role == "solver" else self.config["judge"])
        if seed is not None:
            config["seed"] = seed
        return BoundedLLMClient(config, self.output / "cache" / role,
                                call_seconds=1800, journal=self.output / f"{role}_calls.jsonl")

    def freeze(self):
        paths = sorted({*Path("src/repar").rglob("*.py"),
                        *Path("baselines/BM25_RAG").rglob("*.py"),
                        Path("scripts/evaluate_math.py"), Path("tests/test_math_benchmark.py"),
                        self.config_path, Path(self.config["solver_config"])})
        if "raptor" in self.config["methods"]:
            paths += sorted(Path("baselines/RaptorRAG").rglob("*.py"))
            paths += [Path("baselines/_upstream.py"), Path("baselines/RaptorRAG/sources.json"),
                      Path(self.config["raptor"]["encoder"]) / "download_manifest.json"]
        hashes = {p.as_posix(): file_hash(p) for p in paths}
        identity = {"config": self.config, "resolved_solver": self.solver,
                    "questions_sha256": file_hash(self.config["questions"]),
                    "textbook_sha256": file_hash(self.config["textbook"]),
                    "context_sha256": digest(self.context), "source_hashes": hashes, "rules": self.rules}
        path = self.output / "protocol.json"
        if path.exists():
            if read_json(path)["identity"] != identity:
                raise ValueError("Frozen experiment identity changed: use a new experiment/output, never overwrite historical protocol")
            return
        deployments = {}
        for role, cfg in (("solver", self.solver), ("judge", self.config["judge"])):
            base = cfg["base_url"].rstrip("/")
            models = requests.get(base + "/v1/models", timeout=30).json()
            if cfg["model"] not in [m["id"] for m in models.get("data", [])]:
                raise ValueError(f"Unexpected {role} service model")
            props = requests.get(base + "/props", timeout=30)
            props.raise_for_status()
            deployments[role] = {"models": models, "props": props.json()}
        snapshot = self.output / "source"
        for source in paths:
            target = snapshot / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        write_json(path, {"frozen_at": now(), "identity": identity, "deployments": deployments,
                          "deployment_limit": "service metadata captured; remote judge weight hash not independently measured"})
        (self.output / "context.txt").write_text(self.context, encoding="utf-8")
        write_json(self.output / "context_manifest.json", self.context_manifest)
        print("FROZEN", self.config["experiment_id"], flush=True)

    def calibrate(self):
        self.freeze()
        path = self.output / "calibration.json"
        if path.exists():
            if not read_json(path)["passed"]:
                raise ValueError("Judge calibration failed; inspect preserved evidence")
            return
        client = self.client("judge")
        rows = []
        for name, item, candidate, expected in calibration_cases():
            start = len(client.calls)
            value = grade_once(item, candidate, client, self.config["judge"]["max_tokens"], "calibration_" + name)
            actual = value["units"][0]["credit"]
            rows.append({"name": name, "item": item, "candidate": candidate, "expected": expected,
                         "grade": value, "passed": actual == expected, "calls": client.calls[start:]})
            write_json(path, {"passed": False, "status": "running", "cases": rows})
            print("CALIBRATION", name, actual, "expected", expected, flush=True)
        passed = all(r["passed"] for r in rows)
        write_json(path, {"passed": passed, "status": "complete", "cases": rows})
        if not passed:
            raise ValueError("Judge calibration failed; experiment has not started")

    def direct_context(self, client):
        path = self.output / "direct_context.txt"
        meta_path = self.output / "direct_context.json"
        if path.exists() and meta_path.exists():
            text, meta = path.read_text(encoding="utf-8"), read_json(meta_path)
            if digest(text) != meta["sha256"]:
                raise ValueError("Corrupted direct context")
            return text, meta
        task_reserve = max(count_prompt_tokens(client, solver_messages(task_text(q), ""))
                           for q in self.questions) + 256
        budget = self.solver["direct_input_budget"] - task_reserve
        tokens = client.tokenize(self.context)
        truncated = len(tokens) > budget
        selected = tokens[:budget // 2] + tokens[-(budget - budget // 2):] if truncated else tokens
        text = client.detokenize(selected) if truncated else self.context
        meta = {"original_context_tokens": len(tokens), "retained_context_tokens": len(selected),
                "context_token_budget": budget, "task_and_template_reserve": task_reserve,
                "input_budget": self.solver["direct_input_budget"], "truncated": truncated,
                "policy": "fixed head/tail context, full question preserved, same subset for all questions",
                "sha256": digest(text)}
        path.write_text(text, encoding="utf-8")
        write_json(meta_path, meta)
        return text, meta

    def generate(self, methods=None):
        self.freeze()
        self.calibrate()
        for method in methods or self.config["methods"]:
            path = self.output / f"{method}.jsonl"
            previous = unique_rows(path)
            if len(previous) == len(self.questions):
                continue
            if method not in self.config["methods"]:
                raise ValueError(method)
            client = self.client("solver")
            built = time.perf_counter()
            index, direct_meta = None, None
            if method == "direct":
                direct_text, direct_meta = self.direct_context(client)
            elif method == "bm25":
                index = BM25Retriever(self.context, self.solver["chunk_words"])
            elif method == "raptor":
                from baselines.RaptorRAG import RaptorRetriever
                from repar.model_support import SentenceEncoder

                encoder = SentenceEncoder(self.config["raptor"]["encoder"], threads=2)
                index = RaptorRetriever(self.context, client, encoder,
                                       chunk_words=self.solver["chunk_words"],
                                       **self.config["raptor"]["parameters"])
            else:
                index = KnowledgeIndex.cached(self.context, self.solver, self.output / "indexes")
            write_json(self.output / f"{method}_build.json", {
                "wall_seconds_this_invocation": time.perf_counter() - built,
                "index_stats": index.stats if index else None, "direct": direct_meta})
            for item in self.questions:
                if item["id"] in previous:
                    continue
                started, call_start = time.perf_counter(), len(client.calls)
                row = {"id": item["id"], "method": method, "time": now(), "status": "ok",
                       "question": public_record(item), "context_sha256": digest(self.context),
                       "response": "", "evidence_ids": [], "trace": []}
                task = task_text(item)
                try:
                    if method == "direct":
                        evidence = direct_text
                        row["direct"] = direct_meta
                    elif method == "bm25":
                        evidence, records = retrieve_context(task, index, self.solver["retrieval_top_k"])
                        row.update({"evidence_ids": [r["id"] for r in records], "evidence_records": records})
                    elif method == "raptor":
                        records = index.retrieve(task, self.solver["retrieval_top_k"])
                        if not records:
                            raise ValueError("RAPTOR retrieval returned no evidence")
                        evidence = "\n\n".join(record["text"] for record in records)
                        row.update({"evidence_ids": [r["id"] for r in records], "evidence_records": records})
                    else:
                        result = research(task, index, client, self.solver)
                        evidence = result.evidence
                        row.update({"evidence_ids": result.evidence_ids, "trace": result.trace,
                                    "termination": result.termination, "policies": result.policies})
                    messages = solver_messages(task, evidence or "未取得相关教材内容。")
                    prompt_tokens = count_prompt_tokens(client, messages)
                    if prompt_tokens > self.solver["direct_input_budget"]:
                        raise ValueError("Input exceeds predeclared budget; no silent extra truncation")
                    row["rendered_input_tokens"] = prompt_tokens
                    row["response"] = client.complete(messages, self.solver["answer_max_tokens"], tag=f"math_{method}_answer")
                    if not row["response"].strip():
                        raise ValueError("Empty final response")
                except Exception as exc:
                    row.update({"status": "error", "error": str(exc), "error_traceback": traceback.format_exc()})
                row.update({"seconds": time.perf_counter() - started, "calls": client.calls[call_start:]})
                write_json(self.output / "traces" / method / (item["id"] + ".json"), row)
                append_jsonl(path, row)
                print("GENERATED", method, item["id"], row["status"], f"{row['seconds']:.1f}s",
                      "chars", len(row["response"]), flush=True)
        write_json(self.output / "generation_status.json", {"time": now(), "counts": {
            m: len(unique_rows(self.output / f"{m}.jsonl")) for m in self.config["methods"]}})

    def grade(self):
        self.freeze()
        self.calibrate()
        clients = [self.client("judge", seed) for seed in self.config["judge"]["replicate_seeds"]]
        for method in self.config["methods"]:
            predictions = unique_rows(self.output / f"{method}.jsonl", self.by_id)
            path = self.output / f"{method}_grades.jsonl"
            previous = unique_rows(path)
            for item in self.questions:
                qid = item["id"]
                if qid in previous:
                    continue
                prediction = predictions[qid]
                response = prediction["response"]
                row = {"id": qid, "method": method, "time": now(), "response_sha256": digest(response),
                       "solver_status": prediction["status"], "grades": [], "judge_calls": []}
                if item["question_type"] in {"single_choice", "multiple_choice"}:
                    row.update(score_choice(item, response))
                elif prediction["status"] != "ok":
                    row.update({"fully_correct": False, "score": 0, "strict_score": 0,
                                "max_score": item["score"], "scoring": "solver failure = zero"})
                else:
                    for replicate, client in enumerate(clients):
                        start = len(client.calls)
                        value = grade_once(item, response, client, self.config["judge"]["max_tokens"],
                                           f"math_judge_{qid}_replicate{replicate+1}")
                        row["grades"].append(value)
                        row["judge_calls"].extend(client.calls[start:])
                        write_json(self.output / "judgements" / method / f"{qid}.json", row)
                        if replicate == 1:
                            first, second = row["grades"]
                            a = {u["id"]: u["credit"] for u in first["units"]}
                            b = {u["id"]: u["credit"] for u in second["units"]}
                            if a == b:
                                break
                    row.update(aggregate_grades(item, row["grades"]))
                    row["initial_disagreement"] = len(row["grades"]) == 3
                append_jsonl(path, row)
                print("GRADED", method, qid, f"{row['score']:.4f}/{row['max_score']}",
                      "judge_replicates", len(row["grades"]), flush=True)

    def report(self):
        self.freeze()
        summary, rows = {}, []
        for method in self.config["methods"]:
            predictions = unique_rows(self.output / f"{method}.jsonl", self.by_id)
            grades = unique_rows(self.output / f"{method}_grades.jsonl", self.by_id)
            for qid, g in grades.items():
                if g["response_sha256"] != digest(predictions[qid]["response"]):
                    raise ValueError("Grade does not match prediction")
            qtypes = sorted({q["question_type"] for q in self.questions})
            summary[method] = {"questions": len(grades), "score": sum(g["score"] for g in grades.values()),
                "max_score": 150, "fully_correct": sum(g["fully_correct"] for g in grades.values()),
                "strict_score": sum(g["strict_score"] for g in grades.values()),
                "solver_errors": sum(p["status"] != "ok" for p in predictions.values()),
                "generation_seconds": sum(p["seconds"] for p in predictions.values()),
                "truncated_answers": sum(any(c["response"]["choices"][0].get("finish_reason") == "length"
                                              for c in p["calls"][-1:]) for p in predictions.values()),
                "judge_disagreements": sum(g.get("initial_disagreement", False) for g in grades.values()),
                "judge_review_flags": sum(g.get("review_flag", False) for g in grades.values()),
                "by_type": {t: {"score": sum(g["score"] for qid, g in grades.items() if self.by_id[qid]["question_type"] == t),
                                "max_score": sum(q["score"] for q in self.questions if q["question_type"] == t),
                                "fully_correct": sum(g["fully_correct"] for qid, g in grades.items() if self.by_id[qid]["question_type"] == t),
                                "questions": sum(q["question_type"] == t for q in self.questions)} for t in qtypes}}
            solver_calls = [c for p in predictions.values() for c in p["calls"]]
            judge_calls = [c for g in grades.values() for c in g["judge_calls"]]
            summary[method]["solver_usage"] = usage(solver_calls)
            summary[method]["judge_usage"] = usage(judge_calls)
            summary[method]["tool_calls"] = dict(Counter(e["name"] for p in predictions.values()
                                                       for t in p["trace"] for e in t.get("executed", [])))
            summary[method]["tool_errors"] = sum("error" in e for p in predictions.values()
                                                  for t in p["trace"] for e in t.get("executed", []))
            summary[method]["termination"] = dict(Counter(p.get("termination", "single_pass") for p in predictions.values()))
            summary[method]["controller_repairs"] = sum("_repair" in c["tag"] for c in solver_calls)
            for q in self.questions:
                g, p = grades[q["id"]], predictions[q["id"]]
                rows.append({"id": q["id"], "method": method, "question_type": q["question_type"],
                             "score": g["score"], "max_score": g["max_score"],
                             "fully_correct": g["fully_correct"], "response": p["response"], "grade": g})
        result = {"completed_at": now(), "complete_id_coverage": True, "rules": self.rules, "methods": summary}
        write_json(self.output / "summary.json", result)
        write_json(self.output / "per_question.json", rows)
        lines = ["# MATH 2025 National I 三方法补测", "", f"完成时间：{now()}。19题、150分，57个唯一方法/题号组合已核验。", "",
                 "作答：Qwen3.5-0.8B 标准Q4；评分：9090 的 Qwen3.8-27B-UD-Q4_K_M，非选择题双评、有分歧三评。", "",
                 "**以下为实验折算分，不是官方高考阅卷分。** 多选正确子集按比例给分；解答题等分小问，再乘预定rubric credit。", "",
                 "| 方法 | 实验折算分 /150 | 整题全对 | 严格全对加权分 /150 | 作答失败 |", "|---|---:|---:|---:|---:|"]
        for m, s in summary.items():
            lines.append(f"| {m} | {s['score']:.3f} | {s['fully_correct']}/19 | {s['strict_score']:.1f} | {s['solver_errors']} |")
        lines += ["", "## 分题型", "", "| 方法 | 题型 | 实验分 | 整题全对 |", "|---|---|---:|---:|"]
        for m, s in summary.items():
            for t, v in s["by_type"].items():
                lines.append(f"| {m} | {t} | {v['score']:.3f}/{v['max_score']} | {v['fully_correct']}/{v['questions']} |")
        lines += ["", "## 逐题折算分", "", "| 题号 | 满分 | direct | bm25 | repar |", "|---|---:|---:|---:|---:|"]
        for q in self.questions:
            values = [next(r["score"] for r in rows if r["id"] == q["id"] and r["method"] == m)
                      for m in self.config["methods"]]
            lines.append(f"| {q['question_number']} | {q['score']} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
        lines += ["", "## 限制与复核入口", "",
                  "- 教材全部content字段依文件顺序拼接；不采用原chunk作为检索单元，也不按质量标签过滤。保留OCR噪声，质量分布见context_manifest.json。",
                  "- 直接回答因原文超窗使用固定首尾子集，完整题干保留；检索方法全文建库。此差异限制了方法比较。截断量见direct_context.json。",
                  "- RePAR仍是英语en_core_web_sm抽取器，含拉丁字母筛选，对中文概念/关系覆盖不足；此次未更改核心算法或宣称中文图已支持。",
                  "- 数据及参考答案此前已被人工核对；本次适配时读过参考信息，不能称作未见测试。没有按本次测试分数调参或改prompt。",
                  "- 仅一次确定配置、19题，不能推断稳定优势；相同judge不同seed的双评不是独立人类阅卷，存在相关错误和评分不确定性。",
                  "- 保留所有实际回复、长度截断、控制修复、工具调用和错误；缓存命中时间不作为实测推理加速。协议、源码快照在protocol.json与source/。",
                  "- 每个非选择题的两/三次判分与理由见各方法_grades.jsonl和judgements/，review_flag需人工关注。", "",
                  "## 成本与执行审计", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""]
        (self.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def public_record(item):
    # Reuse the same allowlist for predictions and traces.
    from repar.benchmarks.math_exam import public_question
    return public_question(item)


def count_prompt_tokens(client, messages):
    """Measure model input, not JSON transport escapes (notably OCR newlines)."""
    rendered = client.post("/apply-template", {"messages": messages, "add_generation_prompt": True,
                                               "chat_template_kwargs": {"enable_thinking": client.config["enable_thinking"]}})
    return len(client.tokenize(rendered["prompt"]))


def usage(calls):
    return {"calls": len(calls), "cache_hits": sum(c["cached"] for c in calls),
            "recorded_seconds": sum(c["seconds"] for c in calls),
            "prompt_tokens": sum(c["response"].get("usage", {}).get("prompt_tokens", 0) for c in calls),
            "completion_tokens": sum(c["response"].get("usage", {}).get("completion_tokens", 0) for c in calls)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/math_2025_national_i.json")
    parser.add_argument("--phase", choices=("prepare", "generate", "grade", "report", "all"), default="all")
    parser.add_argument("--method", choices=("direct", "bm25", "repar", "raptor"))
    args = parser.parse_args()
    suite = Suite(args.config)
    if args.phase in {"prepare", "all"}:
        suite.freeze()
        suite.calibrate()
    if args.phase in {"generate", "all"}:
        suite.generate([args.method] if args.method else None)
    if args.phase in {"grade", "all"}:
        suite.grade()
    if args.phase in {"report", "all"}:
        suite.report()


if __name__ == "__main__":
    main()
