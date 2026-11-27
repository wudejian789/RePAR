"""Resume the rerun with each index's own query tokenizer and valid Unicode controls."""
from contextvars import ContextVar
from pathlib import Path

from repar.io import read_json, write_json
from repar.retrieval import lexical
from repar.v4 import selection, workflow

import run_benchmark_recommended as runner
from recover_benchmark_recommended import structured_unicode


_math_research = ContextVar("math_research", default=False)
_prepare = runner.prepare
_case = runner.case
_original_terms = lexical.terms


def prepare(settings, protocols, config):
    data, indexes = _prepare(settings, protocols, config)
    chinese_terms = lexical.terms

    def query_terms(text):
        return chinese_terms(text) if _math_research.get() else _original_terms(text)

    lexical.terms = query_terms

    root = Path(settings["output"])
    for path in root.glob("*/longbench/repar/cases/*.json"):
        row = read_json(path)
        if row.get("query_tokenizer") == "historical":
            continue
        affected = any(
            list(chinese_terms(execution["parameters"].get(key, ""))) != _original_terms(execution["parameters"].get(key, ""))
            for step in row["trace"] for execution in step.get("executed", []) if execution["name"] == "tree_search"
            for key in ("query", "anchor_query")
        )
        if affected:
            condition = path.parent.parent
            archive = condition / "recovery" / path.stem / "tokenizer_scope"
            archive.mkdir(parents=True, exist_ok=True)
            for source, archived_name in ((path, "case.json"),
                                          (condition / "longbench/events" / (path.stem + ".jsonl"), "events.jsonl"),
                                          (condition / "longbench/traces" / path.name, "trace.json")):
                if source.exists():
                    source.replace(archive / archived_name)
            print("REPLAY TOKENIZER", row["id"], flush=True)
    return data, indexes


def one_case(job, data, settings, config, protocol, indexes):
    model, dataset, method, qid, _ = job
    output = Path(settings["output"]) / model / dataset / method
    destination = output / "cases" / (qid + ".json")
    is_new = not destination.exists()
    token = _math_research.set(dataset == "math" and method == "repar")
    try:
        result = _case(job, data, settings, config, protocol, indexes)
    finally:
        _math_research.reset(token)
    if is_new and method == "repar":
        row = read_json(destination)
        row["query_tokenizer"] = "chinese" if dataset == "math" else "historical"
        write_json(destination, row)
    return result


def main():
    settings = read_json(runner.CONFIG_PATH)
    path = Path(settings["output"]) / "resume_implementation.json"
    record = {source: runner.sha(Path(source)) for source in
              ("scripts/resume_benchmark_recommended.py", "scripts/recover_benchmark_recommended.py")}
    if path.exists():
        assert read_json(path) == record
    else:
        write_json(path, record)
    runner.prepare = prepare
    runner.case = one_case
    workflow.structured = selection.structured = structured_unicode
    runner.main()


if __name__ == "__main__":
    main()
