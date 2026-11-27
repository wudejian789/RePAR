"""Ordinary text research tasks; index construction can require LLM calls."""
import argparse
import json
from pathlib import Path
import time

from repar.io import read_json, write_json
from repar.model_support import BoundedLLMClient, SentenceEncoder
from baselines._factory import create_retriever
from baselines._pipeline import answer, retrieve_context


def main(method):
    parser = argparse.ArgumentParser(description=f'{method}: arbitrary knowledge text and free-form tasks')
    parser.add_argument('--context', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--config', required=True, help='LLM config; indexing also uses this model')
    parser.add_argument('--encoder', default='models/all-MiniLM-L6-v2')
    parser.add_argument('--working-dir', required=True, help='An empty or compatible index directory')
    parser.add_argument('--chunk-words', type=int, default=240)
    parser.add_argument('--top-k', type=int, default=24)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--call-seconds', type=float, default=60)
    parser.add_argument('--retrieve-only', action='store_true', help='Skip final generation; construction still calls the LLM')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.seconds <= 0 or args.call_seconds <= 0:
        parser.error('time budgets must be positive')
    work = Path(args.working_dir)
    client = BoundedLLMClient(read_json(args.config), work.parent / (work.name + '_calls'),
        deadline=time.monotonic() + args.seconds, call_seconds=args.call_seconds,
        journal=work.parent / (work.name + '_calls.jsonl'))
    source = create_retriever(method, Path(args.context).read_text(encoding='utf-8'), client,
                              SentenceEncoder(args.encoder), working_dir=work, chunk_words=args.chunk_words)
    try:
        if args.retrieve_only:
            _, records = retrieve_context(args.task, source, args.top_k)
            result = {'evidence_ids': [r['id'] for r in records], 'evidence': records}
        else:
            result = answer(args.task, source, client, args.top_k, args.max_tokens)
        result.update(task=args.task, index_stats=source.stats)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if hasattr(source, 'close'):
            source.close()
