from __future__ import annotations

import asyncio
from pathlib import Path

from repar.model_support import require_unconstrained
from repar.text import split_spans
from baselines._upstream import module, upstream, index_identity, validate_inputs, validate_top_k


class HyperGraphRetriever:
    """Official n-ary extraction and entity/hyperedge hybrid retrieval.

    NetworkX + NanoVectorDB are upstream's default local stores. No Neo4j server
    is needed. A context-only construction interface prevents question leakage.
    """
    def __init__(self, context, client, encoder, *, working_dir, chunk_words=240,
                 extraction_tokens=2048, gleaning=2, workers=1):
        require_unconstrained(client)
        validate_inputs(context, chunk_words)
        if workers < 1:
            raise ValueError('workers must be positive')
        index_identity('HyperGraphRAG', context, client, encoder,
            dict(chunk_words=chunk_words, extraction_tokens=extraction_tokens, gleaning=gleaning, workers=workers), working_dir)
        namespace, self.upstream = upstream("HyperGraphRAG", "hypergraphrag")
        utils = module(namespace, "utils")
        # The official constructor creates a root logger file. Route only that
        # logging callback to the run directory, leaving all algorithms intact.
        core = module(namespace, "hypergraphrag")
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        original_logger = core.set_logger
        core.set_logger = lambda ignored: original_logger(str(self.working_dir / "upstream.log"))

        async def llm(prompt, system_prompt=None, history_messages=None, **kwargs):
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            messages += list(history_messages or [])
            messages.append({"role": "user", "content": prompt})
            if workers == 1:
                return client.complete(messages, kwargs.get("max_tokens", extraction_tokens), tag="hypergraph_infer")
            # Synchronous HTTP inside an async callback previously blocked the
            # upstream extraction event loop. Offload only transport, preserving
            # per-chunk history, gleaning decisions and official prompt content.
            return await asyncio.to_thread(client.complete, messages,
                kwargs.get("max_tokens", extraction_tokens), tag="hypergraph_infer")

        async def embed(texts):
            return encoder.encode(texts)

        try:
            self.engine = core.HyperGraphRAG(working_dir=str(self.working_dir),
                llm_model_func=llm, llm_model_name=client.config["model"],
                llm_model_max_async=workers, embedding_func_max_async=1,
                embedding_func=utils.EmbeddingFunc(embedding_dim=encoder.dimension,
                                                  max_token_size=8192, func=embed),
                entity_extract_max_gleaning=gleaning, enable_llm_cache=False,
                chunk_token_size=1200, chunk_overlap_token_size=0)
        finally:
            core.set_logger = original_logger
        self.query_class = module(namespace, "base").QueryParam
        spans = split_spans(context, chunk_words)
        self.passages = [context[a:b] for a, b in spans]
        self.engine.insert(self.passages)
        graph = self.engine.chunk_entity_relation_graph._graph
        self.stats = {"passages": len(self.passages), "graph_nodes": graph.number_of_nodes(),
                      "graph_edges": graph.number_of_edges(),
                      "hyperedges": sum(str(n).startswith("<hyperedge>") for n in graph),
                      "upstream_commit": self.upstream["commit"]}
        if not self.stats["hyperedges"]:
            raise RuntimeError("Upstream extraction produced no hyperedges; cannot treat this as HyperGraphRAG")

    def retrieve(self, query, top_k=24):
        validate_top_k(top_k)
        # Return the actual upstream evidence bundle (entities, n-ary relations,
        # chunks). Never relabel it as a raw-source passage.
        context = self.engine.query(query, self.query_class(mode="hybrid", top_k=top_k, only_need_context=True))
        if not context or context == module(self.engine.__class__.__module__.rsplit(".", 1)[0], "prompt").PROMPTS["fail_response"]:
            return []
        core = module(self.engine.__class__.__module__.rsplit('.', 1)[0], 'hypergraphrag')
        chunks = core.always_get_an_event_loop().run_until_complete(self.engine.text_chunks.all_keys())
        return [{"id": "HG_CONTEXT", "text": context, "kind": "upstream_context_bundle",
                 "provenance_store": str(self.working_dir / "kv_store_text_chunks.json"),
                 "indexed_chunk_count": len(chunks)}]
