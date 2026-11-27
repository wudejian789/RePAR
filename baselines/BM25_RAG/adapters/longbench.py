"""Preserve the existing LongBench BM25 input, answer parser and request tag."""
from repar.evaluation import extract_answer, official_prompt, qa_text

from ..pipeline import retrieve_context
from ..retriever import BM25Retriever


def answer(item, index, client, config):
    source = index if index is not None else BM25Retriever(item["context"], config["chunk_words"])
    context, records = retrieve_context(qa_text(item), source, config["retrieval_top_k"])
    response = client.complete([{"role": "user", "content": official_prompt(item, context, rag=True)}],
                               config["answer_max_tokens"], tag="bm25_answer")
    return {"response": response, "pred": extract_answer(response),
            "evidence_ids": [record["id"] for record in records],
            "index_stats": source.stats, "context_sha256": source.context_hash}
