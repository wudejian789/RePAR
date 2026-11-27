"""Token-bounded source partitions and a dense ordered tree for v4-r60."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from functools import lru_cache
from hashlib import file_digest
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import pickle
import re
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

from .index_v2 import KnowledgeIndex
from .io import digest


R60 = "react-v4-r60"
R61 = "react-v4-r61"
V5 = "react-v5"
FOCUSED_DENSE_VERSIONS = {R61, V5}
DENSE_VERSIONS = {R60} | FOCUSED_DENSE_VERSIONS
INDEX_VERSION = "embeddinggemma-tree-entity-v1"


@lru_cache(maxsize=2)
def load_encoder(path, device, manifest_sha256):
    from sentence_transformers import SentenceTransformer
    import torch

    root = Path(path)
    manifest = root / "source_manifest.json"
    with manifest.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != manifest_sha256:
            raise ValueError("EmbeddingGemma manifest differs from the index configuration")
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    for name, entry in files.items():
        with (root / name).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f"EmbeddingGemma file differs from its manifest: {name}")
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    return SentenceTransformer(path, device=device, local_files_only=True)


@lru_cache(maxsize=4096)
def query_vector(path, device, manifest_sha256, text):
    return load_encoder(path, device, manifest_sha256).encode_query(
        text, normalize_embeddings=True, show_progress_bar=False)


def split_token_spans(text, tokenizer, max_tokens):
    """Cut the untouched text using fast tokenizer offsets and nearby sentence boundaries."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("chunk_max_tokens must be a positive integer")
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)["offset_mapping"]
    starts = [start for start, end in offsets if start < end]
    paragraphs = [m.end() for m in re.finditer(r"\n[ \t\r]*\n+", text)]
    sentences = [m.end() for m in re.finditer(r"[。！？]|[.!?](?=\s|$)", text)]
    spans, counts = [], []
    start, first = 0, 0
    while start < len(text):
        limit = first + max_tokens
        end = len(text) if limit >= len(starts) else starts[limit]
        if end < len(text) and end > start:
            floor = starts[first + 2 * max_tokens // 3]
            for boundaries in (paragraphs, sentences):
                i = bisect_right(boundaries, end) - 1
                if i >= 0 and start < boundaries[i] >= floor:
                    end = boundaries[i]
                    break
        if end <= start:
            end = start + 1
        while True:
            count = len(tokenizer(text[start:end], add_special_tokens=False, truncation=False,
                                  verbose=False)["input_ids"])
            if count <= max_tokens:
                break
            if end == start + 1:
                raise ValueError("chunk_max_tokens cannot encode an indivisible character")
            end -= 1
        spans.append((start, end))
        counts.append(count)
        start = end
        first = bisect_left(starts, end, first)
    return (spans, counts) if spans else ([(0, 0)], [0])


class DenseKnowledgeIndex(KnowledgeIndex):
    @classmethod
    def cache_path(cls, context, config, directory):
        identity = {key: config[key] for key in ("chunk_max_tokens", "tree_fanout", "index_version", "graph_model")}
        identity["embedding"] = {key: config["embedding"].get(key) for key in ("manifest_sha256", "build_device", "batch_size")}
        for name in ("spacy", "en-core-web-sm", "zh-core-web-sm"):
            try:
                identity[name] = version(name)
            except PackageNotFoundError:
                identity[name] = None
        return Path(directory)/(digest(context + json.dumps(identity, sort_keys=True)) + ".pkl")

    @classmethod
    def cached(cls, context, config, directory="artifacts/indexes_r60", *, build=True):
        path = cls.cache_path(context, config, directory)
        if path.is_file():
            with path.open("rb") as stream:
                index = pickle.load(stream)
            if index.context != context or index.stats["chunk_max_tokens"] != config["chunk_max_tokens"]:
                raise ValueError(f"Cached r60 context or chunk budget differs: {path}")
            index.config = config
            return index
        if not build:
            raise FileNotFoundError(path)
        index = cls(context, config)
        index.save(path)
        return index

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            pickle.dump(self, stream, pickle.HIGHEST_PROTOCOL)
        try:
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    def encoder_settings(self, *, build=False):
        cfg = self.config["embedding"]
        device = cfg.get("build_device", "cpu") if build else cfg.get("query_device", "cpu")
        return str(Path(cfg["path"]).resolve()), device, cfg["manifest_sha256"]

    def __init__(self, context, config, *, build_graph=True):
        started = perf_counter()
        self.context, self.context_hash, self.config = context, digest(context), config
        encoder = load_encoder(*self.encoder_settings(build=True))
        limit = config["chunk_max_tokens"]
        self.spans, self.chunk_tokens = split_token_spans(context, encoder.tokenizer, limit)
        self.texts = [context[a:b] for a, b in self.spans]
        prompt = encoder.prompts["document"]
        for start in range(0, len(self.texts), 256):
            batch = self.texts[start:start+256]
            lengths = encoder.tokenizer([prompt + text.strip() for text in batch], truncation=False,
                                        verbose=False, return_length=True)["length"]
            if max(lengths) > encoder.max_seq_length:
                raise ValueError("Document prefix and source tokens exceed the embedding model window")
        pre_encode = perf_counter()
        self.matrix = encoder.encode_document(self.texts, normalize_embeddings=True,
            batch_size=config["embedding"].get("batch_size", 32), show_progress_bar=False)
        encoding_seconds = perf_counter() - pre_encode
        if not np.isfinite(self.matrix).all():
            raise ValueError("EmbeddingGemma returned invalid source vectors")
        self.build_tree(config["tree_fanout"])
        self.knowledge = [{"concepts": [], "logic_cues": []} for _ in self.texts]
        self.stats = {"context_sha256": self.context_hash, "characters": len(context), "leaves": len(self.spans),
                      "tree_nodes": len(self.nodes), "version": config["index_version"],
                      "build_seconds": perf_counter()-started, "embedding_seconds": encoding_seconds,
                      "chunk_max_tokens": limit, "observed_max_chunk_tokens": max(self.chunk_tokens),
                      "embedding_dimensions": self.matrix.shape[1],
                      "embedding_manifest_sha256": config["embedding"]["manifest_sha256"]}
        if build_graph:
            self._build_graph()

    def build_tree(self, fanout):
        if type(fanout) is not int or fanout < 2:
            raise ValueError("tree_fanout must be an integer of at least two")
        self.nodes, self.leaf_to_tree = [], {}
        vectors = []
        adjacency = np.einsum("ij,ij->i", self.matrix[:-1], self.matrix[1:])

        def build(lo, hi, parent=None):
            nid = len(self.nodes)
            node = {"id": f"T{nid}", "lo": lo, "hi": hi, "parent": parent, "children": [], "keywords": []}
            self.nodes.append(node)
            vectors.append(None)
            if hi-lo == 1:
                vectors[nid] = self.matrix[lo]
                self.leaf_to_tree[lo] = nid
                return nid
            count = min(fanout, hi-lo)
            cuts = [lo]
            for j in range(1, count):
                center = lo+(hi-lo)*j//count
                radius = max(1, (hi-lo)//(count*3))
                candidates = range(max(cuts[-1]+1, center-radius), min(hi-(count-j), center+radius)+1)
                cuts.append(min(candidates, key=lambda cut: (adjacency[cut-1], abs(cut-center))))
            cuts.append(hi)
            node["children"] = [build(a, b, nid) for a, b in zip(cuts, cuts[1:])]
            vector = self.matrix[lo:hi].mean(axis=0)
            vectors[nid] = vector / max(float(np.linalg.norm(vector)), 1e-12)
            return nid

        build(0, len(self.texts))
        self.tree_matrix = np.stack(vectors)

    def tree_search(self, query, top_k=8, scope=None, beam_width=8, mode="hybrid"):
        if mode not in {"tree", "hybrid"}:
            raise ValueError("mode must be tree or hybrid")
        top_k = max(1, min(24, int(top_k)))
        width = max(top_k, min(32, int(beam_width)))
        start = self._scope(scope)
        node = self.nodes[start]
        leaves, visited = [], 0
        text = str(query).strip()
        candidates = {"method": "embeddinggemma_cosine", "routed_ids": [], "exact_dense_ids": []}
        if text:
            vector = query_vector(*self.encoder_settings(), text)
            active = [start]
            while active:
                visited += len(active)
                expanded = []
                for nid in active:
                    branch = self.nodes[nid]
                    if branch["children"]:
                        expanded.extend(branch["children"])
                    else:
                        leaves.append(branch["lo"])
                scores = self.tree_matrix[expanded] @ vector
                ranked = np.lexsort((expanded, -scores))[:width]
                active = [expanded[i] for i in ranked]
            lo, hi = node["lo"], node["hi"]
            similarity = self.matrix[lo:hi] @ vector
            exact_limit = top_k if getattr(self, "config", {}).get("agent_version") in FOCUSED_DENSE_VERSIONS else min(2, top_k)
            exact = (np.argsort(-similarity, kind="stable")[:exact_limit] + lo).tolist() if mode == "hybrid" else []
            candidates.update(routed_ids=[f"C{i}" for i in leaves], exact_dense_ids=[f"C{i}" for i in exact])
            chosen = sorted(set(leaves+exact), key=lambda i: (-similarity[i-lo], i))[:top_k]
            results = [{**self.record(i), "score": float(similarity[i-lo])} for i in chosen if similarity[i-lo] > 0]
        else:
            results = []
        return {"operator": "tree_search", "visited_tree_nodes": visited, "scope": node["id"], "mode": mode,
                "results": results, "candidate_generation": candidates}
