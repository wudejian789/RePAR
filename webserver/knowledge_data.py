"""Export the structures used by archived retrieval runs."""
from __future__ import annotations

import pickle

from repar.io import digest
from repar.text import split_spans

from .frozen_data import file_digest


def bm25_data(context, stats):
    spans = split_spans(context, stats["chunk_words"])
    if len(spans) != stats["leaves"] or digest(context) != stats["context_sha256"]:
        raise ValueError("BM25 partition differs from the recorded index")
    return {"kind": "bm25", "chunk_words": stats["chunk_words"],
            "chunks": [{"id": f"C{i}", "start": start, "end": end, "text": context[start:end]}
                       for i, (start, end) in enumerate(spans)]}


def load_frozen_index(root, name, references, sources):
    path = root / name
    expected = references[name]
    if path not in sources:
        actual = file_digest(path)
        if actual != expected:
            raise ValueError(f"Frozen index hash mismatch: {path}")
        sources[path] = actual
    elif sources[path] != expected:
        raise ValueError(f"Conflicting frozen index hashes: {path}")
    with path.open("rb") as stream:
        return pickle.load(stream)


def tree_graph_data(index, context):
    if index.context != context or index.context_hash != digest(context):
        raise ValueError("Frozen index source mismatch")
    chunks = []
    graph = getattr(index, "entity_graph", None)
    for i, (start, end) in enumerate(index.spans):
        if index.texts[i] != context[start:end]:
            raise ValueError(f"Frozen index block C{i} differs from source")
        chunks.append({"id": f"C{i}", "start": start, "end": end, "text": index.texts[i],
                       "tree": f"T{index.leaf_to_tree[i]}", "concepts": index.knowledge[i]["concepts"],
                       "entity_ids": graph.passage_entities[i] if graph else [],
                       "proposition_ids": graph.passage_propositions[i] if graph else [],
                       "claims": index.knowledge[i].get("claims", [])})
    if graph:
        for record in graph.propositions:
            if context[record["start"]:record["end"]] != record["text"]:
                raise ValueError(f"Frozen proposition {record['id']} differs from source")
        return {"kind": "entity", "chunks": chunks, "nodes": index.nodes,
                "extractor": graph.extractor, "entities": graph.entities,
                "propositions": graph.propositions, "relations": graph.relations}
    relations = [
        {**edge, "id": f"C{i}:{j}", "subject": f"C{i}", "object": f"C{edge['target']}",
         "predicate": edge["relation"], "kind": "passage_association", "passage_ids": [f"C{i}", f"C{edge['target']}"]}
        for i, edges in enumerate(index.edges) for j, edge in enumerate(edges)
    ]
    return {"kind": "passage", "chunks": chunks, "nodes": index.nodes, "relations": relations}
