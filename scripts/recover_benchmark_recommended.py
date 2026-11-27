"""Recover invalid Unicode in recorded control responses for the benchmark rerun."""
import json
from pathlib import Path

import jsonschema

from repar.io import read_json, read_jsonl, write_json
from repar.v3.workflow import StructuredOutputError
from repar.v4 import selection, workflow

from run_benchmark_recommended import CONFIG_PATH, OLD, case, preflight, prepare


def structured_unicode(client, messages, schema, budget, tag):
    for attempt in range(2):
        raw = client.complete(messages, budget, json_mode=True, schema=schema,
                              tag=tag + ("_repair" if attempt else ""))
        try:
            value = json.loads(raw)
            json.dumps(value, ensure_ascii=False).encode("utf-8")
            jsonschema.validate(value, schema)
            return value
        except (ValueError, TypeError, jsonschema.ValidationError) as error:
            if attempt:
                raise StructuredOutputError(tag, raw) from error
            messages = [*messages, {"role": "assistant", "content": raw.encode("utf-8", "backslashreplace").decode("utf-8")},
                        {"role": "user", "content": "Return a complete concise object satisfying the schema."}]


def main():
    settings = read_json(CONFIG_PATH)
    protocols = {m: read_json(p / "protocol.json") for m, p in OLD.items()}
    config = dict(protocols["08"]["identity"]["config"], longbench_final_max_tokens=settings["longbench"]["max_tokens"],
                  longbench_answer_grammar=settings["longbench"]["grammar"])
    preflight(settings, protocols, config)
    root = Path(settings["output"])
    work = []
    for path in root.glob("*/*/repar/*/traces/*.json"):
        model, dataset, method = path.relative_to(root).parts[:3]
        case_path = path.parents[2] / "cases" / path.name
        if case_path.exists():
            continue
        prior = read_json(path)
        if prior.get("status") == "error" and "surrogates not allowed" in prior.get("error", ""):
            work.append((model, dataset, method, path, prior))
    if not work:
        return
    data, indexes = prepare(settings, protocols, config)
    workflow.structured = selection.structured = structured_unicode
    for model, dataset, method, path, prior in work:
        qid = path.stem
        output = path.parents[2]
        recovery = output / "recovery" / qid
        write_json(recovery / "original_trace.json", prior)
        event = path.parent.parent / "events" / (qid + ".jsonl")
        if event.exists():
            write_json(recovery / "original_events.json", read_jsonl(event))
            event.unlink()
        if model == "08" and dataset == "longbench":
            urls = settings["longbench_08_parallel_endpoints"]
            endpoint = urls[list(data[dataset]).index(qid) % len(urls)]
        else:
            endpoint = protocols[model]["identity"]["solver"]["base_url"]
        job, status = case((model, dataset, method, qid, endpoint), data, settings, config, protocols, indexes)
        write_json(recovery / "result.json", {"job": job, "status": status, "reason": "invalid Unicode in JSON control output"})
        print("RECOVERED", *job[:4], status, flush=True)


if __name__ == "__main__":
    main()
