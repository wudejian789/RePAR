from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path):
    if not Path(path).exists():
        return []
    # JSON strings may contain literal U+2028/U+2029. Only physical JSONL
    # newline delimiters separate records; str.splitlines() is too broad.
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def load_public_samples(path):
    """Labels never leave the evaluator; index and agent receive only public fields."""
    data = read_json(path)
    return [x for x in data if x["domain"].lower().replace("-", " ").startswith("multi")]


def public_item(item):
    return {k: v for k, v in item.items() if k != "answer"}
