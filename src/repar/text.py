"""Shared source segmentation and lexical normalization, independent of agents."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def terms(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}|\d+(?:\.\d+)?|[\u4e00-\u9fff]", text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS]


def split_spans(text, target_words=240):
    """Ordered, lossless partition; prefer paragraph then sentence boundaries."""
    words = list(re.finditer(r"\S+", text))
    if not words:
        return [(0, len(text))]
    spans, start, wi = [], 0, 0
    while wi < len(words):
        end_word = min(len(words), wi + target_words)
        end = words[end_word].start() if end_word < len(words) else len(text)
        if end < len(text):
            floor = words[min(len(words) - 1, wi + target_words * 2 // 3)].start()
            boundary = text.rfind("\n\n", floor, end)
            if boundary >= floor:
                end = boundary + 2
            else:
                boundaries = list(re.finditer(r"[.!?]\s+(?=[A-Z])", text[floor:end]))
                if boundaries:
                    end = floor + boundaries[-1].end()
        if end <= start:
            end = len(text)
        spans.append((start, end))
        start = end
        while wi < len(words) and words[wi].start() < end:
            wi += 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans
