"""Answer a free-form task against a UTF-8 knowledge source using RePAR v5."""
import argparse
from pathlib import Path

from repar.dense_index import DenseKnowledgeIndex
from repar.io import read_json, write_json
from repar.model_support import BoundedLLMClient
from repar.v5 import answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", default="configs/repar_v5.json")
    parser.add_argument("--output", default="artifacts/research_v5.json")
    args = parser.parse_args()
    config = read_json(args.config)
    context = args.context.read_text(encoding="utf-8-sig")
    index = DenseKnowledgeIndex.cached(context, config, directory="artifacts/indexes_v5")
    client = BoundedLLMClient(config, "artifacts/cache/research_v5", call_seconds=1800)
    events = Path(args.output + ".events.jsonl")
    events.unlink(missing_ok=True)
    result = answer(args.task, index, client, config, event_path=events)
    write_json(args.output, result)
    print(result["response"])


if __name__ == "__main__":
    main()
