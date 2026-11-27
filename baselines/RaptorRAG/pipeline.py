from __future__ import annotations

import numpy as np

from repar.model_support import require_unconstrained
from repar.text import split_spans
from baselines._upstream import module, upstream, validate_inputs, validate_top_k


class RaptorRetriever:
    """Use upstream UMAP + soft GMM + recursive summaries + collapsed retrieval.

    Only context enters construction. Original punctuation and offsets are kept;
    unlike upstream split_text, shared lossless splitting never rewrites evidence.
    """
    def __init__(self, context, client, encoder, *, chunk_words=240, num_layers=5,
                 summary_tokens=100, seed=42, max_context_tokens=8000,
                 guard_small_clusters=False):
        require_unconstrained(client)
        validate_inputs(context, chunk_words)
        if num_layers < 1 or summary_tokens < 1 or max_context_tokens < 1:
            raise ValueError('layer and token budgets must be positive')
        namespace, self.upstream = upstream("RaptorRAG", "raptor")
        embedding_base = module(namespace, "EmbeddingModels").BaseEmbeddingModel
        summary_base = module(namespace, "SummarizationModels").BaseSummarizationModel
        builders = module(namespace, "cluster_tree_builder")
        structures = module(namespace, "tree_structures")
        retrievers = module(namespace, "tree_retriever")
        import tiktoken
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = max_context_tokens
        self.spans = split_spans(context, chunk_words)
        self.context = context

        class Embedding(embedding_base):
            def create_embedding(self, text):
                return encoder.encode([text])[0].tolist()

        class Summary(summary_base):
            def summarize(self, context, max_tokens=100):
                # Same task-independent instruction as the upstream summarizer.
                return client.complete([{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content":
                     f"Write a summary of the following, including as many key details as possible: {context}:"}],
                    max_tokens, tag="raptor_summary")

        self.embedding = Embedding()
        config = builders.ClusterTreeConfig(tokenizer=self.tokenizer, max_tokens=400,
            num_layers=num_layers, summarization_length=summary_tokens, summarization_model=Summary(),
            embedding_models={"LOCAL": self.embedding}, cluster_embedding_model="LOCAL")
        small_cluster_splits = 0
        if guard_small_clusters:
            clustering = module(namespace, "cluster_utils")

            class SmallClusterSafeRaptor:
                @staticmethod
                def perform_clustering(nodes, embedding_model_name, max_length_in_cluster=3500,
                                       tokenizer=self.tokenizer, reduction_dimension=10,
                                       threshold=0.1):
                    nonlocal small_cluster_splits
                    if len(nodes) <= 4:
                        small_cluster_splits += 1
                        return [[node] for node in nodes]
                    vectors = np.array([node.embeddings[embedding_model_name] for node in nodes])
                    labels = clustering.perform_clustering(
                        vectors, dim=reduction_dimension, threshold=threshold)
                    groups = []
                    for label in np.unique(np.concatenate(labels)):
                        group = [node for node, assigned in zip(nodes, labels) if label in assigned]
                        if len(group) == 1 or sum(len(tokenizer.encode(node.text)) for node in group) <= max_length_in_cluster:
                            groups.append(group)
                        elif len(group) == len(nodes):
                            groups.extend([node] for node in group)
                        else:
                            groups.extend(SmallClusterSafeRaptor.perform_clustering(
                                group, embedding_model_name, max_length_in_cluster,
                                tokenizer, reduction_dimension, threshold))
                    return groups

            config.clustering_algorithm = SmallClusterSafeRaptor
        builder = builders.ClusterTreeBuilder(config)
        texts = [context[a:b] for a, b in self.spans]
        vectors = encoder.encode(texts)
        leaves = {i: structures.Node(text, i, set(), {"LOCAL": vectors[i].tolist()})
                  for i, text in enumerate(texts)}
        layers = {0: list(leaves.values())}
        nodes = dict(leaves)
        np.random.seed(seed)
        # Official stopping criterion and clustering algorithm remain unchanged.
        roots = builder.construct_tree(dict(leaves), nodes, layers, use_multithreading=False)
        self.tree = structures.Tree(nodes, roots, leaves, builder.num_layers, layers)
        rc = retrievers.TreeRetrieverConfig(tokenizer=self.tokenizer, embedding_model=self.embedding,
                                           context_embedding_model="LOCAL")
        self.retriever = retrievers.TreeRetriever(rc, self.tree)
        self.stats = {"leaves": len(leaves), "nodes": len(nodes), "layers": len(layers),
                      "summaries": len(nodes) - len(leaves), "upstream_commit": self.upstream["commit"]}

        if guard_small_clusters:
            self.stats["small_cluster_splits"] = small_cluster_splits

    def _leaves(self, node_id):
        node = self.tree.all_nodes[node_id]
        if not node.children:
            return [node_id]
        return sorted({leaf for child in node.children for leaf in self._leaves(child)})

    def retrieve(self, query, top_k=24):
        validate_top_k(top_k)
        selected, _ = self.retriever.retrieve_information_collapse_tree(query, top_k, self.max_context_tokens)
        return [{"id": f"R{node.index}", "text": node.text, "kind": "summary" if node.children else "source",
                 "source_spans": [{"start": self.spans[i][0], "end": self.spans[i][1]}
                                  for i in self._leaves(node.index)]} for node in selected]
