from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path

from .client import LLMClient
from .evaluation import direct
from .io import append_jsonl, digest, load_public_samples, public_item, read_json, read_jsonl, write_json


def run(args):
    config = read_json(args.config)
    samples = load_public_samples(args.data)
    if args.limit:
        samples = samples[:args.limit]
    output = Path(args.output) / f"{args.method}.jsonl"
    done = {x["_id"] for x in read_jsonl(output)}
    config_hash = digest(__import__("json").dumps(config, sort_keys=True))
    metadata_path = Path(args.output) / "run_metadata.json"
    if metadata_path.exists():
        assert read_json(metadata_path)["config_sha256"] == config_hash, "Cannot mix different configs"
    else:
        write_json(metadata_path, {"config": config, "config_sha256": config_hash,
                                  "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    output_lock = Lock()

    def process(entry):
        i, labelled = entry
        if (Path(args.output) / (args.method + ".stop")).exists():
            print(f"Graceful stop requested for {args.method}; resume after removing the .stop file.", flush=True)
            return
        if labelled["_id"] in done:
            return
        item = public_item(labelled)
        client = LLMClient(config)
        started = time.perf_counter()
        offline_wait_seconds = 0
        print(f"START {args.method} {i+1}/{len(samples)} {item['_id']} chars={len(item['context'])}", flush=True)
        try:
            if args.method == "direct":
                result = direct(item, client, config)
            elif args.method == "bm25":
                from baselines.BM25_RAG.adapters.longbench import answer as baseline_answer
                result = baseline_answer(item, None, client, config)
            else:
                if config.get("agent_version") == "react-v2":
                    from .agent_v2 import answer
                    from .index_v2 import KnowledgeIndex
                    if getattr(args,"wait_for_indexes",False):
                        import json
                        key=digest(item["context"]+json.dumps({k:config.get(k) for k in
                            ("chunk_words","tree_fanout","graph_model","index_version")},sort_keys=True))
                        ready=Path("artifacts/indexes_v2")/(key+".pkl")
                        deadline=time.monotonic()+7200
                        wait_started=time.perf_counter()
                        while not ready.exists():
                            if time.monotonic()>deadline:
                                raise RuntimeError("Timed out waiting for the independent knowledge index build")
                            time.sleep(3)
                        offline_wait_seconds=time.perf_counter()-wait_started
                else:
                    from .agent import answer
                    from .index import KnowledgeIndex
                index = KnowledgeIndex.cached(item["context"], config)
                result = answer(item, index, client, config, args.method)
            result.update({k: labelled[k] for k in ("_id", "domain", "sub_domain", "length", "difficulty", "answer")})
            result.update({"judge": result["pred"] == labelled["answer"],
                           "method": args.method, "seconds": time.perf_counter() - started - offline_wait_seconds,
                           "offline_index_wait_seconds": offline_wait_seconds,
                           "config_sha256": config_hash,
                           "calls": [{k: v for k, v in c.items() if k != "response"} |
                                     {"usage": c["response"].get("usage", {}),
                                      "timings": c["response"].get("timings", {}),
                                      "finish_reason": c["response"]["choices"][0].get("finish_reason")}
                                     for c in client.calls]})
            with output_lock:
                append_jsonl(output, result)
            # Report progress without inspecting correctness during development.
            print(f"DONE {args.method} {i+1}/{len(samples)} seconds={result['seconds']:.1f} calls={len(client.calls)}", flush=True)
        except Exception as exc:
            with output_lock:
                append_jsonl(Path(args.output) / "errors.jsonl", {
                    "_id": item["_id"], "method": args.method, "error": str(exc),
                    "traceback": traceback.format_exc(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            raise

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(process, enumerate(samples)))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--method", choices=["direct", "bm25", "repar", "repar_tree"], required=True)
    r.add_argument("--data", default="data/longbench_v2.json")
    r.add_argument("--config", default="configs/repar_v2.json")
    r.add_argument("--output", default="artifacts/experiment_v2")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--workers", type=int, default=1, help="Independent question requests; each trajectory remains sequential")
    r.add_argument("--wait-for-indexes",action="store_true",help="Wait for the independent v2 index builder instead of constructing a missing graph")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
