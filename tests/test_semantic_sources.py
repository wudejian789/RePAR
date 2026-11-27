"""Question-free source-vector cache, raw span mapping and scoped retrieval."""
from pathlib import Path
import hashlib

import numpy as np
import pytest

from repar.v4.semantic import SemanticSources


class SourceIndex:
    texts = ['Aster Library keeps the register.', 'Juniper Museum holds the painting.', 'Maple Archive stores the notebooks.']
    context = '\n\n'.join(texts)
    context_hash = hashlib.sha256(context.encode('utf-8')).hexdigest()
    spans = [(0, len(texts[0])), (len(texts[0])+2, len(texts[0])+2+len(texts[1])),
             (len(texts[0])+len(texts[1])+4, len(context))]
    nodes = [{'lo': 0, 'hi': 3}, {'lo': 1, 'hi': 2}]
    def _scope(self, scope): return 1 if scope == 'T1' else 0
    def record(self, i, score):
        return {'id': f'C{i}', 'text': self.texts[i], 'start': self.spans[i][0], 'end': self.spans[i][1], 'score': score}


class Similarity:
    manifest = {'model': 'fixture', 'revision': 'r1'}
    def __init__(self): self.requests = []
    def vectors(self, texts, prefix):
        self.requests.append((list(texts), prefix))
        if prefix == 'passage: ':
            return np.eye(3, dtype=np.float32)
        return np.asarray([[0.1, 0.9, 0.2] for _ in texts], dtype=np.float32)


def test_cache_construction_depends_only_on_original_source_texts(tmp_path):
    index, similarity = SourceIndex(), Similarity()
    cache = SemanticSources(index, similarity, tmp_path)
    assert similarity.requests == [(index.texts, 'passage: ')]
    assert cache.metadata['source_text_or_partitions_changed'] is False
    assert not cache.metadata['reused']
    rows = cache.search('Where is the painting?', 2)
    assert [r['id'] for r in rows] == ['C1', 'C2']
    assert rows[0]['text'] == index.texts[1]
    assert cache.metadata['query_calls'] == 1 and cache.metadata['query_seconds'] >= 0


def test_scope_is_respected_and_identical_source_cache_reuses_vectors(tmp_path):
    index = SourceIndex()
    initial = SemanticSources(index, Similarity(), tmp_path)
    other = Similarity()
    cache = SemanticSources(index, other, tmp_path)
    assert cache.metadata['reused'] and other.requests == []
    assert cache.metadata['sha256'] == initial.metadata['sha256']
    assert [r['id'] for r in cache.search('notebooks', 8, 'T1')] == ['C1']


def test_source_exclusion_refills_the_requested_semantic_rank_inside_scope(tmp_path):
    cache = SemanticSources(SourceIndex(), Similarity(), tmp_path)
    assert [r['id'] for r in cache.search('painting', 1, exclude_passages={'C1'})] == ['C2']
    assert cache.search('painting', 1, 'T1', exclude_passages={'C1'}) == []


def test_changed_source_mapping_or_encoder_revision_uses_new_identity(tmp_path):
    original = SemanticSources(SourceIndex(), Similarity(), tmp_path)
    index = SourceIndex(); index.spans = [(1, index.spans[0][1]), *index.spans[1:]]
    index.texts = [index.texts[0][1:], *index.texts[1:]]
    altered = SemanticSources(index, Similarity(), tmp_path)
    scorer = Similarity(); scorer.manifest = {'model': 'fixture', 'revision': 'r2'}
    another = SemanticSources(SourceIndex(), scorer, tmp_path)
    assert len({x.metadata['path'] for x in [original, altered, another]}) == 3


def test_corrupt_vector_cache_is_rejected(tmp_path):
    initial = SemanticSources(SourceIndex(), Similarity(), tmp_path)
    path = Path(initial.metadata['path'])
    with np.load(path, allow_pickle=False) as data:
        identity, spans = str(data['identity']), data['spans']
    np.savez(path, identity=identity, spans=spans, vectors=np.asarray([[np.nan]*3]*3))
    with pytest.raises(ValueError, match='Invalid semantic source-cache'):
        SemanticSources(SourceIndex(), Similarity(), tmp_path)


def test_mismatched_original_context_identity_is_rejected(tmp_path):
    index = SourceIndex(); index.context_hash = 'unrelated-source-hash'
    with pytest.raises(ValueError, match='original-context hash differs'):
        SemanticSources(index, Similarity(), tmp_path)


def test_concurrent_cache_builders_keep_one_immutable_canonical_file(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    barrier = threading.Barrier(2)
    class Concurrent(Similarity):
        def vectors(self, texts, prefix):
            barrier.wait(timeout=10)
            return super().vectors(texts, prefix)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(SemanticSources, SourceIndex(), Concurrent(), tmp_path) for _ in range(2)]
        results = [f.result(timeout=20) for f in futures]
    assert results[0].metadata['sha256'] == results[1].metadata['sha256']
    assert len(list(tmp_path.glob('*.npz'))) == 1 and not list(tmp_path.glob('*.tmp'))
    assert np.array_equal(results[0].vectors, results[1].vectors)
