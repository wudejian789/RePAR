"""Task-independent query identity and lexical query anchoring."""
from __future__ import annotations

import json
import re
import unicodedata


def request_identity(name, parameters):
    def normalize(value):
        if isinstance(value, str):
            return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()
        if isinstance(value, list):
            return [normalize(x) for x in value]
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in sorted(value.items())}
        return value
    return json.dumps([name, normalize(parameters)], ensure_ascii=False, sort_keys=True)


def search_query(focus, task):
    """Keep task anchors alongside the focus without rewriting source text."""
    if not focus.strip() or focus.strip() == task.strip():
        return task
    return task + "\n" + focus
