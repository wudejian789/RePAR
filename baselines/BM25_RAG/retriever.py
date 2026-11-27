"""A source-only BM25 index; builds neither a semantic tree nor an entity graph."""
from __future__ import annotations

import time

from repar.io import digest
from repar.retrieval import build_postings, rank_bm25
from repar.text import split_spans


class BM25Retriever:
    def __init__(self, context, chunk_words=240):
        if chunk_words < 1:
            raise ValueError("chunk_words must be positive")
        started = time.perf_counter()
        self.context = context
        self.context_hash = digest(context)
        self.spans = split_spans(context, chunk_words)
        self.texts = [context[a:b] for a, b in self.spans]
        self.postings, self.lengths, self.avg_length = build_postings(self.texts)
        self.stats = {"context_sha256": self.context_hash, "characters": len(context),
                      "leaves": len(self.spans), "build_seconds": time.perf_counter()-started,
                      "version": "bm25-rag-v1", "chunk_words": chunk_words}

    def bm25(self, query, top_k=24):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        return rank_bm25(query, self.postings, self.lengths, self.avg_length, top_k)

    def record(self, i, score=0):
        start, end = self.spans[i]
        return {"id": f"C{i}", "start": start, "end": end, "text": self.texts[i], "score": round(score, 5)}
