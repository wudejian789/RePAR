"""Shared BM25 operator; contains no RAG workflow or benchmark conventions."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from ..text import terms


def build_postings(texts):
    postings = defaultdict(list)
    lengths = np.zeros(len(texts), dtype=np.float32)
    for i, text in enumerate(texts):
        counts = Counter(terms(text))
        lengths[i] = sum(counts.values())
        for term, count in counts.items():
            postings[term].append((i, count))
    avg_length = max(1, float(lengths.mean())) if len(lengths) else 1
    return postings, lengths, avg_length


def bm25_idf(document_count, frequency):
    return math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))


def rank_bm25(query, postings, lengths, avg_length, top_k=24, lo=0, hi=None):
    """Rank a source range using global corpus statistics and stable source-order ties.

    Preserve the original operator's k1=1.5, b=0.75 and capped query term
    frequency, so moving the baseline does not alter historical request inputs.
    """
    scores = np.zeros(len(lengths), dtype=np.float64)
    n = len(scores)
    for term, qtf in Counter(terms(query)).items():
        posting = postings.get(term, [])
        idf = bm25_idf(n, len(posting))
        for i, tf in posting:
            scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[i] / avg_length)) * min(qtf, 2)
    ids = np.arange(lo, n if hi is None else hi)
    return [(int(i), float(scores[i])) for i in sorted(ids, key=lambda i: (-scores[i], int(i)))[:top_k]]
