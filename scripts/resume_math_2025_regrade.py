"""Resume the 27B choice judge and verify quotations joined with ellipses."""
import json
from pathlib import Path
import re

from repar.io import read_json, write_json

import regrade_math_2025 as grading
from run_benchmark_recommended import sha


_original_observation = grading.choice_observation


def choice_observation(question, answer, raw):
    try:
        return _original_observation(question, answer, raw)
    except ValueError as error:
        value = json.loads(raw)
        quote = value.get("quote")
        if not isinstance(quote, str) or len(quote) > 240:
            raise error
        parts = re.split(r"\.{3,}|…+", quote)
        if len(parts) == 1:
            raise error
        candidate = grading.normalize_quote(answer)
        cursor = 0
        for part in parts:
            normalized = grading.normalize_quote(part)
            start = candidate.find(normalized, cursor)
            if not normalized or start < 0:
                raise error
            cursor = start + len(normalized)
        observation = _original_observation(question, answer, json.dumps({**value, "quote": parts[0]}))
        return {**observation, "quote": quote}


def recovery_manifest(*, create=False):
    path = grading.ROOT / "quote_validation.json"
    identity = {"implementation_sha256": sha(Path(__file__)), "initial_protocol_sha256": sha(grading.ROOT / "protocol.json"),
                "rule": "each excerpt separated by an ellipsis must occur in order in the candidate answer"}
    if path.exists():
        assert read_json(path) == identity
    else:
        assert create
        write_json(path, identity)


if __name__ == "__main__":
    recovery_manifest(create=True)
    grading.choice_observation = choice_observation
    grading.main()
