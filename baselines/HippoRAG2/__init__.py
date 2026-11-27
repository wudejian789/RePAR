"""Pinned official HippoRAG 2 engine with local model injection."""
from .pipeline import HippoRetriever
from baselines._pipeline import answer, retrieve_context

__all__ = ["HippoRetriever", "answer", "retrieve_context"]
