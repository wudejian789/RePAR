"""Retrieval primitives shared by agents and independent baselines."""
from .lexical import build_postings, rank_bm25

__all__ = ["build_postings", "rank_bm25"]
