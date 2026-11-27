"""Run the baseline on arbitrary UTF-8 text without any benchmark fields."""
import argparse
import json
from pathlib import Path

from repar.client import LLMClient
from repar.io import read_json, write_json

from .pipeline import answer, retrieve_context
from .retriever import BM25Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", help="Model connection/config JSON; required for answer generation")
    parser.add_argument("--chunk-words", type=int, default=240)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retrieve-only", action="store_true", help="Read evidence without calling a model")
    parser.add_argument("--output", help="Optional JSON result path")
    args = parser.parse_args()
    if not args.retrieve_only and not args.config:
        parser.error("--config is required unless --retrieve-only is specified")
    source = BM25Retriever(Path(args.context).read_text(encoding="utf-8"), args.chunk_words)
    if args.retrieve_only:
        _, records = retrieve_context(args.task, source, args.top_k)
        result = {"evidence_ids": [r["id"] for r in records], "evidence": records}
    else:
        config = read_json(args.config)
        config.pop("answer_grammar", None)  # generic tasks are not constrained to choice letters
        result = answer(args.task, source, LLMClient(config), args.top_k, args.max_tokens)
    result.update({"task": args.task, "context_sha256": source.context_hash, "index_stats": source.stats})
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
