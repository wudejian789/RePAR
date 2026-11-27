from __future__ import annotations

import math
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .io import digest, write_json
from .retrieval import build_postings, rank_bm25
# Preserve these public names for existing callers and pickled vectorizers.
from .text import terms, split_spans


class KnowledgeIndex:
    """Question-independent ordered semantic tree + provenance-preserving proposition graph."""

    @classmethod
    def cached(cls, context, config, directory="artifacts/indexes"):
        key = digest(context + str({k: config[k] for k in
                     ("chunk_words", "tree_fanout", "graph_neighbors", "index_version")}))
        path = Path(directory) / (key + ".pkl")
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        index = cls(context, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
        write_json(path.with_suffix(".stats.json"), index.stats)
        return index

    def __init__(self, context, config):
        started = time.perf_counter()
        self.context = context
        self.context_hash = digest(context)
        self.config = config
        self.spans = split_spans(context, config["chunk_words"])
        self.texts = [context[a:b] for a, b in self.spans]
        self.vectorizer = TfidfVectorizer(tokenizer=terms, token_pattern=None, lowercase=False,
                                        ngram_range=(1, 2), max_features=60000, sublinear_tf=True,
                                        dtype=np.float32)
        try:
            self.matrix = self.vectorizer.fit_transform(self.texts).tocsr()
        except ValueError:
            self.matrix = self.vectorizer.fit_transform(["empty " + t for t in self.texts]).tocsr()
        self.features = self.vectorizer.get_feature_names_out()
        self.nodes, vectors = [], []
        self.leaf_to_tree = {}
        adjacency = np.asarray(self.matrix[:-1].multiply(self.matrix[1:]).sum(axis=1)).ravel()

        def build(lo, hi, parent=None):
            nid = len(self.nodes)
            node = {"id": f"T{nid}", "lo": lo, "hi": hi, "parent": parent, "children": []}
            self.nodes.append(node)
            vectors.append(None)
            if hi - lo == 1:
                row = self.matrix[lo]
                self.leaf_to_tree[lo] = nid
            else:
                fanout = min(config["tree_fanout"], hi - lo)
                cuts = [lo]
                # Weak adjacent semantic links near balanced quantiles define ordered splits.
                for j in range(1, fanout):
                    center = lo + (hi - lo) * j // fanout
                    radius = max(1, (hi - lo) // (fanout * 3))
                    a = max(cuts[-1] + 1, center - radius)
                    b = min(hi - (fanout - j), center + radius)
                    cut = min(range(a, b + 1), key=lambda c: (adjacency[c - 1], abs(c - center)))
                    cuts.append(cut)
                cuts.append(hi)
                children = [build(a, b, nid) for a, b in zip(cuts, cuts[1:])]
                node["children"] = children
                row = vectors[children[0]].copy()
                for child in children[1:]:
                    row = row + vectors[child]
                row = normalize(row).tocsr()
            vectors[nid] = row
            order = np.argsort(row.data)[-10:][::-1]
            node["keywords"] = [str(self.features[row.indices[k]]) for k in order]
            return nid

        build(0, len(self.texts))
        self.tree_matrix = sparse.vstack(vectors, format="csr")
        self.knowledge = []
        for i, text in enumerate(self.texts):
            row = self.matrix[i]
            order = np.argsort(row.data)[-12:][::-1]
            keywords = [str(self.features[row.indices[k]]) for k in order]
            sentences = [m for m in re.finditer(r"[^.!?]+(?:[.!?](?=\s|$)|$)", text)
                         if len(m.group().split()) >= 7]
            def salience(m):
                return sum(float(row[0, self.vectorizer.vocabulary_[w]]) for w in set(terms(m.group()))
                           if w in self.vectorizer.vocabulary_) / math.sqrt(max(1, len(m.group().split())))
            selected = sorted(sorted(sentences, key=salience, reverse=True)[:3], key=lambda m: m.start())
            claims = [{"text": m.group().strip(), "start": self.spans[i][0] + m.start(),
                       "end": self.spans[i][0] + m.end()} for m in selected]
            cues = {"causal_cue": r"\bbecause\b|\btherefore\b|\bresult(?:ed|s)? in\b|\bdue to\b",
                    "contrast_cue": r"\bhowever\b|\bunlike\b|\bwhereas\b|\bin contrast\b",
                    "temporal_cue": r"\bbefore\b|\bafter\b|\bduring\b|\b(?:19|20)\d{2}\b"}
            self.knowledge.append({"id": f"K{i}", "leaf": i, "concepts": keywords,
                                   "claims": claims, "logic_cues": [k for k, p in cues.items()
                                                                             if re.search(p, text, re.I)]})
        self.edges = [[] for _ in self.texts]
        # Sparse all-pairs blocks avoid an unbounded dense N*N allocation.
        for start in range(0, len(self.texts), 128):
            sims = (self.matrix[start:start + 128] @ self.matrix.T).tocsr()
            for local in range(sims.shape[0]):
                i = start + local
                row = sims[local]
                candidates = sorted(zip(row.indices, row.data), key=lambda x: (-float(x[1]), int(x[0])))
                count = 0
                for j, score in candidates:
                    j = int(j)
                    if i == j or abs(i - j) <= 1 or score < 0.05:
                        continue
                    common = sorted(set(self.knowledge[i]["concepts"]) & set(self.knowledge[j]["concepts"]))
                    relation = "shared_concept" if common else "semantic_related"
                    self.edges[i].append({"target": j, "relation": relation, "score": float(score),
                                          "bridge": common[:4], "provenance": [f"C{i}", f"C{j}"]})
                    count += 1
                    if count >= config["graph_neighbors"]:
                        break
                for j in (i - 1, i + 1):
                    if 0 <= j < len(self.texts):
                        self.edges[i].append({"target": j, "relation": "sequence", "score": 0.15,
                                              "bridge": [], "provenance": [f"C{i}", f"C{j}"]})
        # A bounded shared-entity index adds bridges beyond the closest lexical neighbors.
        entity_postings = defaultdict(list)
        for i, text in enumerate(self.texts):
            entities = set(re.findall(r"\b[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3}\b|\b[A-Z]{3,}\b", text))
            for entity in entities:
                entity_postings[entity].append(i)
        for entity, ids in entity_postings.items():
            if 2 <= len(ids) <= 24:
                for pos, i in enumerate(ids):
                    for j in ids[max(0, pos - 2):pos] + ids[pos + 1:pos + 3]:
                        if abs(i - j) > 1:
                            self.edges[i].append({"target": j, "relation": "shared_entity", "score": 0.3,
                                                  "bridge": [entity], "provenance": [f"C{i}", f"C{j}"]})
        # BM25 uses the same leaves as all retrieval variants.
        self.counts = Counter()
        self.postings, self.lengths, self.avg_length = build_postings(self.texts)
        self.stats = {"context_sha256": self.context_hash, "characters": len(context),
                      "leaves": len(self.spans), "tree_nodes": len(self.nodes),
                      "knowledge_nodes": len(self.knowledge), "edges": sum(map(len, self.edges)),
                      "build_seconds": time.perf_counter() - started, "version": config["index_version"]}

    def _leaf_id(self, value):
        i = int(re.sub(r"^[CK]", "", str(value)))
        if not 0 <= i < len(self.texts):
            raise ValueError(f"Unknown evidence ID: {value}")
        return i

    def _scope(self, scope):
        if scope in (None, "", "root"):
            return 0
        n = int(str(scope).removeprefix("T"))
        if not 0 <= n < len(self.nodes):
            raise ValueError(f"Unknown tree scope: {scope}")
        return n

    def bm25(self, query, top_k=24, scope=None):
        node = self.nodes[self._scope(scope)]
        return rank_bm25(query, self.postings, self.lengths, self.avg_length, top_k, node["lo"], node["hi"])

    def record(self, i, score=0, preview=False):
        a, b = self.spans[i]
        path, n = [], self.leaf_to_tree[i]
        while n is not None:
            path.append(self.nodes[n]["id"])
            n = self.nodes[n]["parent"]
        return {"id": f"C{i}", "knowledge_id": f"K{i}", "tree_path": path[::-1],
                "start": a, "end": b, "score": round(score, 5),
                "concepts": self.knowledge[i]["concepts"][:8],
                "logic_cues": self.knowledge[i]["logic_cues"],
                "text": self.texts[i][:500] if preview else self.texts[i]}

    def tree_search(self, query, top_k=8, scope=None, beam_width=8, mode="hybrid"):
        top_k = max(1, min(24, int(top_k)))
        q = self.vectorizer.transform([str(query)])
        start = self._scope(scope)
        active, leaves, visited = [start], [], 0
        while active:
            expanded = []
            for n in active:
                visited += 1
                node = self.nodes[n]
                if node["children"]:
                    expanded.extend(node["children"])
                else:
                    leaves.append(node["lo"])
            if not expanded:
                break
            scores = np.asarray((self.tree_matrix[expanded] @ q.T).toarray()).ravel()
            ranked = sorted(zip(expanded, scores), key=lambda x: (-float(x[1]), x[0]))
            active = [n for n, _ in ranked[:max(top_k, min(32, int(beam_width)))]]
        ranked = self.bm25(str(query), max(64, top_k * 4), scope)
        rank = {i: s for i, s in ranked}
        # Sparse inverted-list candidates protect recall when centroid routing is ambiguous.
        candidates = set(leaves)
        if mode == "hybrid":
            candidates.update(i for i, _ in ranked[:min(2, top_k)])
        chosen = sorted(candidates, key=lambda i: (-rank.get(i, 0), i))[:top_k]
        return {"operator": "tree_search", "visited_tree_nodes": visited,
                "scope": self.nodes[start]["id"], "mode": mode,
                "results": [self.record(i, rank.get(i, 0)) for i in chosen]}

    def tree_browse(self, node="T0", top_k=12):
        n = self._scope(node)
        data = self.nodes[n]
        children = data["children"] or [n]
        return {"operator": "tree_browse", "node": data,
                "children": [self.nodes[c] | {"preview": self.texts[self.nodes[c]["lo"]][:240]}
                             for c in children[:max(1, min(24, int(top_k)))]]}

    def read(self, ids, radius=0):
        radius = max(0, min(2, int(radius)))
        chosen = set()
        for value in ids[:16]:
            i = self._leaf_id(value)
            chosen.update(range(max(0, i - radius), min(len(self.texts), i + radius + 1)))
        return {"operator": "read", "results": [self.record(i) for i in sorted(chosen)[:24]]}

    def graph_search(self, query="", seeds=None, hops=1, top_k=8, relations=None):
        top_k = max(1, min(24, int(top_k)))
        if seeds:
            seed_ids = [self._leaf_id(x) for x in seeds[:12]]
        else:
            seed_ids = [i for i, _ in self.bm25(str(query), 4)]
        q = self.vectorizer.transform([str(query)])
        qscore = np.asarray((self.matrix @ q.T).toarray()).ravel()
        frontier = {i: 1.0 for i in seed_ids}
        visited = set(seed_ids)
        candidates, paths = {}, {}
        for depth in range(max(1, min(3, int(hops)))):
            next_frontier = {}
            for i, path_score in frontier.items():
                for edge in self.edges[i]:
                    j = edge["target"]
                    if relations and edge["relation"] not in relations:
                        continue
                    if j in visited:
                        continue
                    score = 0.65 * float(qscore[j]) + 0.35 * path_score * edge["score"] * 0.8 ** depth
                    if score > candidates.get(j, -1):
                        candidates[j] = score
                        paths[j] = {"from": f"K{i}", "to": f"K{j}", "depth": depth + 1,
                                    **{k: v for k, v in edge.items() if k != "target"}}
                        next_frontier[j] = score
            visited.update(next_frontier)
            frontier = dict(sorted(next_frontier.items(), key=lambda x: (-x[1], x[0]))[:24])
        ranked = sorted(candidates, key=lambda i: (-candidates[i], i))[:top_k]
        return {"operator": "graph_search", "seeds": [f"K{i}" for i in seed_ids],
                "results": [self.record(i, candidates[i]) | {"edge": paths[i]} for i in ranked]}

    def execute(self, name, parameters, allow_graph=True):
        functions = {"tree_search": self.tree_search, "tree_browse": self.tree_browse, "read": self.read}
        if allow_graph:
            functions["graph_search"] = self.graph_search
        if name not in functions:
            raise ValueError(f"Unknown or disabled operator: {name}")
        return functions[name](**parameters)
