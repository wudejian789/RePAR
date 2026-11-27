"""Rerun benchmark answers against frozen indexes and archived baseline evidence."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from hashlib import file_digest
from pathlib import Path
import pickle
import sys
from time import perf_counter

import requests

from repar.benchmarks.math_exam import public_question, stitch_context, task_text
from repar.benchmarks.v3_suite import Suite
from repar.benchmarks.v4_suite import V4Suite
from repar.evaluation import extract_answer, qa_text
from repar.io import digest, load_public_samples, public_item, read_json, read_jsonl, write_json
from repar.model_support import BoundedLLMClient


CONFIG_PATH = Path("configs/benchmark_recommended_20261123.json")
ARTIFACTS = Path("artifacts")
OLD = {
    "08": ARTIFACTS / "repar_v4_r27_20261120",
    "27": ARTIFACTS / "repar_v4_r27_27b_20261120",
}
BASELINE_METHODS = {"raptor": "RaptorRAG", "hipporag2": "HippoRAG2", "hypergraphrag": "HyperGraphRAG"}
MATH_08 = ARTIFACTS / "math_2025_national_i_v5"
EXTRA = ARTIFACTS / "baseline_full_qwen08_dual_v1"
CHINESE = ARTIFACTS / "repar_v4_20261123_zh_index"


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def historical_rows(model, dataset, method):
    if model == "27":
        root = ARTIFACTS / "baselines_27b_20261119" / method
        source = root / dataset / "predictions.jsonl"
        return {row["id"]: (row, root / "cache/solver") for row in read_jsonl(source)}
    if dataset == "math":
        root = ARTIFACTS / "math_2025_national_i_raptor_v2" if method == "raptor" else MATH_08
        return {row["id"]: (row, root / "cache/solver") for row in read_jsonl(root / (method + ".jsonl"))}
    if method in {"direct", "bm25"}:
        source = ARTIFACTS / "experiment" / (method + ".jsonl")
        return {row["_id"]: (row, ARTIFACTS / "cache/llm") for row in read_jsonl(source)}
    upstream = BASELINE_METHODS[method]
    rows = [row for row in read_json(EXTRA / "per_question.json") if row["method"] == upstream]
    result = {}
    for row in rows:
        case = EXTRA / upstream / row["id"]
        journal = case / "attempts" / Path(row["attempt"]).name / "calls.jsonl"
        calls = [call for call in read_jsonl(journal) if call.get("tag") == "baseline_longbench_answer"
                 and call.get("event") == "completion"] if journal.exists() else []
        result[row["id"]] = ({**row, "calls": calls}, case / "cache")
    assert len(result) == len(rows)
    return result


def archived_request(source, expected_hash):
    row, cache = source
    assert row["context_sha256"] == expected_hash
    if not row["calls"]:
        if row["status"] in {"ok", "completed"}:
            raise ValueError("Completed baseline lacks a final request")
        return None, None
    path = cache / (row["calls"][-1]["key"] + ".json")
    return read_json(path)["request"], path.as_posix()


def check_final(body, config, settings, model, dataset):
    assert body["model"] == config["model"] and body["seed"] == config["seed"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    if dataset == "math":
        assert body["max_tokens"] == settings["math_answer_max_tokens"]
        assert "grammar" not in body and "response_format" not in body
        assert all(body[key] == value for key, value in settings["sampling"][model].items())
    else:
        assert all(body[key] == value for key, value in settings["longbench"].items())
        assert "response_format" not in body


def baseline_answer(qid, source, context_hash, solver, output, settings, model, dataset):
    previous, cache_path = archived_request(source, context_hash)
    row = {"id": qid, "context_sha256": context_hash, "calls": [], "response": "", "seconds": None,
           "historical_request": cache_path, "evidence_reuse": "archived_final_input"}
    if previous is None:
        historical = source[0]
        return {**row, "status": "historical_index_failure", "evidence_reuse": "unavailable",
                "historical_phase": historical.get("phase"), "historical_error": historical.get("error")}
    assert previous["model"] == solver["model"]
    body = dict(previous)
    if dataset == "math":
        body.update(settings["sampling"][model], max_tokens=settings["math_answer_max_tokens"])
    else:
        body.update(settings["longbench"])
    check_final(body, solver, settings, model, dataset)
    client = BoundedLLMClient(solver, output / "cache/solver", call_seconds=1200)
    started = perf_counter()
    call = client._complete_body(body, "final_answer")
    return {**row, "status": "ok", "response": call["response"]["choices"][0]["message"].get("content") or "",
            "calls": client.calls, "seconds": perf_counter() - started}


class RecommendedSuite(V4Suite):
    def __init__(self, solver, config, output):
        self.solver = solver
        self.config = config
        self.output = output

    def client(self, role="solver", seed=None):
        config = dict(self.solver if role == "solver" else self.config["judge"])
        if seed is not None:
            config["seed"] = seed
        return BoundedLLMClient(config, self.output / "cache" / role, call_seconds=1200,
                                journal=self.output / f"{role}_calls.jsonl" if role == "judge" else None)


def case(job, data, settings, config, protocol, indexes):
    model, dataset, method, qid, endpoint = job
    output = Path(settings["output"]) / model / dataset / method
    dest = output / "cases" / (qid + ".json")
    if dest.exists():
        return job, read_json(dest)["status"]
    solver = dict(protocol[model]["identity"]["solver"])
    solver.update(base_url=endpoint, deployment_id=f"{settings['experiment_id']}/{endpoint}")
    if dataset == "math":
        solver.update(sampling=settings["sampling"][model], answer_max_tokens=settings["math_answer_max_tokens"])
    elif endpoint in settings["longbench_08_parallel_endpoints"]:
        solver["context_window"] = 65536
    question = data[dataset][qid]
    context_hash = data["math_hash"] if dataset == "math" else digest(question["context"])
    if method == "repar":
        suite = RecommendedSuite(solver, config, output)
        if dataset == "math":
            index, reuse = indexes[model]
            row = suite.generate(qid, task_text(question), index, dataset)
            row["question"] = public_question(question)
        else:
            item = public_item(question)
            started = perf_counter()
            index, reuse = suite.load_index(item["context"], dataset)
            load_seconds = perf_counter() - started
            row = suite.generate(qid, qa_text(item), index, dataset, longbench_item=item)
            row["index_load_wall_seconds"] = load_seconds
        row["index_reuse"] = reuse
        if row["status"] != "ok":
            raise RuntimeError(f"{model}/{dataset}/{method}/{qid}: {row.get('error')}")
        raw = read_json(output / "cache/solver" / (row["calls"][-1]["key"] + ".json"))["request"]
        check_final(raw, solver, settings, model, dataset)
    else:
        source = data["sources"][(model, dataset, method)][qid]
        row = baseline_answer(qid, source, context_hash, solver, output, settings, model, dataset)
    if dataset == "longbench":
        row["pred"] = extract_answer(row["response"])
        row["gold"] = question["answer"]
        row["correct"] = row["pred"] == row["gold"]
    row.update(model=model, dataset=dataset, method=method, base_url=endpoint)
    write_json(dest, row)
    return job, row["status"]


def preflight(settings, protocol, config):
    root = Path(settings["output"])
    root.mkdir(parents=True, exist_ok=True)
    expected = {model: value["deployments"]["solver"]["props"] for model, value in protocol.items()}
    extra = read_json(EXTRA / "protocol.json")["serving"]
    endpoints = {protocol[m]["identity"]["solver"]["base_url"]: props for m, props in expected.items()}
    endpoints.update({url: extra[url]["props"] for url in settings["longbench_08_parallel_endpoints"]})
    identity = {"configuration": settings,
                "files": {p.as_posix(): sha(p) for p in
                          [CONFIG_PATH, Path(__file__), Path("third_party/pred.py"),
                           *(OLD[m] / "protocol.json" for m in OLD), Path(settings["chinese_index"]),
                           *(Path(config[key]) for key in ("math_questions", "math_textbook", "longbench_data"))]},
                "core": {p.as_posix(): sha(p) for p in sorted(Path("src/repar").rglob("*.py"))}}
    props = {}
    for endpoint, conditions in endpoints.items():
        response = requests.get(endpoint + "/props", timeout=15)
        response.raise_for_status()
        props[endpoint] = response.json()
        for key in ("model_alias", "build_info", "total_slots", "default_generation_settings"):
            assert props[endpoint][key] == conditions[key], (endpoint, key)
    path = root / "protocol.json"
    if path.exists():
        assert read_json(path)["identity"] == identity
    else:
        write_json(path, {"identity": identity, "deployments": props, "judge": config["judge"],
                          "frozen_at": datetime.now().astimezone().isoformat()})


def prepare(settings, protocol, config):
    sys.path[:0] = [str(CHINESE.resolve()), str((CHINESE / "deps").resolve())]
    from zh_index import parser, set_query_tokenizer

    context, meta = stitch_context(config["math_textbook"])
    samples = load_public_samples(config["longbench_data"])
    data = {"math": {q["id"]: q for q in read_jsonl(config["math_questions"])},
            "longbench": {q["_id"]: q for q in samples}, "math_hash": meta["sha256"], "sources": {}}
    assert len(data["math"]) == 19 and len(data["longbench"]) == 125
    for group, methods in settings["conditions"].items():
        model, dataset = group.split("/")
        for method in methods:
            if method != "repar":
                source = historical_rows(model, dataset, method)
                assert set(source) == set(data[dataset]), (group, method)
                data["sources"][(model, dataset, method)] = source
    filename = Path(settings["chinese_index"])
    with filename.open("rb") as stream:
        index = pickle.load(stream)
    assert index.context == context and index.stats["extractor"]["model"] == "zh_core_web_sm"
    parser()
    set_query_tokenizer()
    reference = {"path": filename.as_posix(), "sha256": sha(filename), "reused": True, "stats": index.stats}
    return data, {model: (index, reference) for model in protocol}


def run(settings, protocol, config, select):
    data, indexes = prepare(settings, protocol, config)
    queues = {}
    dual = settings["longbench_08_parallel_endpoints"]
    for group, methods in settings["conditions"].items():
        if select and group not in select:
            continue
        model, dataset = group.split("/")
        base = protocol[model]["identity"]["solver"]["base_url"]
        for method in methods:
            for offset, qid in enumerate(data[dataset]):
                endpoint = dual[offset % len(dual)] if model == "08" and dataset == "longbench" and method != "direct" else base
                queues.setdefault(endpoint, []).append((model, dataset, method, qid, endpoint))
    pools = []
    try:
        pending = []
        for url, jobs in queues.items():
            workers = settings["longbench_08_workers_per_endpoint"] if url in dual else (settings["workers_27"] if url.endswith(":9090") else 2)
            pool = ThreadPoolExecutor(max_workers=workers)
            pools.append(pool)
            pending.extend(pool.submit(case, job, data, settings, config, protocol, indexes) for job in jobs)
        errors = []
        for future in as_completed(pending):
            try:
                job, status = future.result()
                print("DONE", *job[:4], status, flush=True)
            except Exception as error:
                print("ERROR", repr(error), flush=True)
                errors.append(repr(error))
        if errors:
            raise RuntimeError(f"{len(errors)} cases need retry. First errors: {errors[:4]}")
    finally:
        for pool in pools:
            pool.shutdown(cancel_futures=True)


def grade(settings, protocol, config):
    for group, methods in settings["conditions"].items():
        model, dataset = group.split("/")
        if dataset != "math":
            continue
        expected = [row["id"] for row in read_jsonl(config["math_questions"])]
        for method in methods:
            output = Path(settings["output"]) / model / dataset / method
            if any(not (output / "cases" / (qid + ".json")).exists() for qid in expected):
                continue
            destination = output / "math/predictions.jsonl"
            destination.parent.mkdir(exist_ok=True)
            rows = [read_json(output / "cases" / (qid + ".json")) for qid in expected]
            import json

            body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            if destination.exists():
                assert destination.read_text(encoding="utf-8") == body
            else:
                destination.write_text(body, encoding="utf-8")
            suite = RecommendedSuite(protocol[model]["identity"]["solver"], config, output)
            Suite.grade_math(suite)
            print("GRADED", model, method, flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("generate", "grade", "all"))
    cli.add_argument("--group", action="append", choices=("08/math", "08/longbench", "27/math", "27/longbench"))
    args = cli.parse_args()
    settings = read_json(CONFIG_PATH)
    protocol = {model: read_json(folder / "protocol.json") for model, folder in OLD.items()}
    config = dict(protocol["08"]["identity"]["config"], longbench_final_max_tokens=settings["longbench"]["max_tokens"],
                  longbench_answer_grammar=settings["longbench"]["grammar"])
    preflight(settings, protocol, config)
    if args.phase in ("generate", "all"):
        run(settings, protocol, config, args.group)
    if args.phase in ("grade", "all"):
        grade(settings, protocol, config)


if __name__ == "__main__":
    main()
