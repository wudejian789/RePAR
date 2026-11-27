from __future__ import annotations

from repar.model_support import require_unconstrained
from repar.text import split_spans
from baselines._upstream import module, upstream, index_identity, validate_inputs, validate_top_k


class HippoRetriever:
    """Official OpenIE, recognition filtering, synonym links and PPR."""
    def __init__(self, context, client, encoder, *, working_dir, chunk_words=240,
                 ner_tokens=2048, triple_tokens=4096, workers=1):
        require_unconstrained(client)
        validate_inputs(context, chunk_words)
        if workers < 1:
            raise ValueError('workers must be positive')
        identity = index_identity('HippoRAG2', context, client, encoder,
            dict(chunk_words=chunk_words, ner_tokens=ner_tokens, triple_tokens=triple_tokens, workers=workers), working_dir)
        namespace, self.upstream = upstream("HippoRAG2", "src/hipporag")
        config_class = module(namespace, "utils.config_utils").BaseConfig
        llms = module(namespace, "llm.base")
        embedding_base = module(namespace, "embedding_model.base").BaseEmbeddingModel
        engine_class = module(namespace, "HippoRAG").HippoRAG
        self.spans = split_spans(context, chunk_words)
        self.passages = [context[a:b] for a, b in self.spans]
        self._by_text = {text: i for i, text in enumerate(self.passages)}
        cfg = config_class(save_dir=str(working_dir), llm_name=client.config["model"],
            llm_base_url=client.base + "/v1", embedding_model_name=encoder.identity,
            openie_mode="online", openie_max_workers=workers, max_retry_attempts=0,
            openie_ner_max_tokens=ner_tokens, openie_triple_max_tokens=triple_tokens,
            temperature=client.config["temperature"], seed=client.config["seed"],
            preprocess_chunk_max_token_size=None)

        class LocalLLM(llms.BaseLLM):
            def _init_llm_config(self):
                self.llm_config = llms.LLMConfig.from_dict({"model": cfg.llm_name})

            def infer(self, messages=None, chat=None, **kwargs):
                limit = kwargs.get("max_new_tokens", kwargs.get("max_tokens", cfg.max_new_tokens))
                text = client.complete(messages if messages is not None else chat, limit,
                    json_mode=bool(kwargs.get("response_format")), tag="hipporag_infer")
                call = client.last_call if hasattr(client, 'last_call') else client.calls[-1]
                response = call["response"]
                return text, {**response.get("usage", {}),
                    "finish_reason": response["choices"][0].get("finish_reason")}, call["cached"]

        class LocalEmbedding(embedding_base):
            def batch_encode(self, texts, **kwargs):
                # The upstream Transformers adapter also omits instructions for SBERT.
                return encoder.encode(texts)

        llm = LocalLLM(cfg)
        llm._init_llm_config()
        embedding = LocalEmbedding(cfg)
        embedding.embedding_dim = encoder.dimension
        self.engine = engine_class(global_config=cfg, extraction_llm=llm, qa_llm=llm,
                                   embedding_model=embedding, index_identity=identity)
        # No queries or gold labels are passed to the upstream indexer.
        chunk_class = module(namespace, 'utils.misc_utils').Chunk
        docs = [chunk_class(content=text, source_id=f"C{i}",
                 metadata={"start": self.spans[i][0], "end": self.spans[i][1]})
                for i, text in enumerate(self.passages)]
        self.engine.index(docs=docs)
        self.stats = {"passages": len(docs), "graph_nodes": self.engine.graph.vcount(),
                      "graph_edges": self.engine.graph.ecount(), "upstream_commit": self.upstream["commit"]}

    def retrieve(self, query, top_k=24):
        validate_top_k(top_k)
        result = self.engine.retrieve([query], num_to_retrieve=top_k)[0]
        self.stats.update(graph_seeds=result.graph_seeds, facts=len(self.engine.fact_node_keys))
        self.stats['retrieval_mode'] = 'ppr' if result.graph_seeds else 'upstream_dense_fallback'
        records = []
        for text, score in zip(result.docs, result.doc_scores):
            i = self._by_text[text]
            records.append({"id": f"C{i}", "text": text, "score": float(score),
                            "start": self.spans[i][0], "end": self.spans[i][1], "kind": "source"})
        return records

    def close(self):
        self.engine.close()
