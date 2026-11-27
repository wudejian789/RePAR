"""Optional frozen CPU encoder for comparing queries with literal source text.

This module neither generates answers nor constructs or changes KB indexes.
Long inputs are represented by all bounded token windows, rather than silently
discarding their suffix. Encoding windows never replace the original evidence.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import os
import tempfile

import numpy as np


@lru_cache(maxsize=2)
def load_encoder(directory, expected_manifest, threads):
    path = Path(directory)
    raw = (path/'download_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest:
        raise ValueError('Retrieval encoder manifest differs from configured identity')
    manifest = json.loads(raw)
    for record in manifest['files']:
        if hashlib.sha256((path/record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Frozen retrieval encoder file changed: '+record['path'])
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(threads)
    return SentenceTransformer(str(path), device='cpu', local_files_only=True), manifest


class SourceSimilarity:
    def __init__(self, config):
        started = time.perf_counter()
        self.model, self.manifest = load_encoder(config['path'], config['manifest_sha256'], config.get('threads', 2))
        self.load_seconds = time.perf_counter()-started
        self.cache = OrderedDict()
        self.cache_limit = config.get('cache_items', 8192)
        self.batch_size = config.get('batch_size', 32)
        self.calls, self.cache_hits, self.texts_encoded, self.encoding_windows, self.long_inputs = 0, 0, 0, 0, 0
        self.seconds = 0.0

    def pieces(self, text, prefix):
        # Prefix is part of the encoder's documented retrieval interface.
        ids = self.model.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        capacity = self.model.max_seq_length-8
        if len(ids) <= capacity:
            return [prefix+text]
        self.long_inputs += 1
        stride = max(1, capacity*3//4)
        starts = list(range(0, len(ids)-capacity+1, stride))
        if starts[-1] != len(ids)-capacity:
            starts.append(len(ids)-capacity)
        return [prefix+self.model.tokenizer.decode(ids[start:start+capacity], skip_special_tokens=True)
                for start in starts]

    def vectors(self, texts, prefix):
        keys = [(prefix, text) for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        self.cache_hits += len(keys)-len(missing)
        if missing:
            windows, spans = [], []
            for _, text in missing:
                start = len(windows)
                windows.extend(self.pieces(text, prefix))
                spans.append((start, len(windows)))
            vectors = self.model.encode(windows, batch_size=self.batch_size,
                normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            self.encoding_windows += len(windows)
            self.texts_encoded += len(missing)
            for key, (start, end) in zip(missing, spans):
                vector = vectors[start:end].mean(axis=0)
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                self.cache[key] = vector
        # Materialize the answer before eviction, including a very large batch.
        result = np.stack([self.cache[key] for key in keys])
        for key in keys:
            self.cache.move_to_end(key)
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return result

    def scores(self, query, texts):
        if not texts:
            return []
        started = time.perf_counter()
        self.calls += 1
        query_vector = self.vectors([query], 'query: ')[0]
        vectors = self.vectors(texts, 'passage: ')
        result = [float(value) for value in vectors @ query_vector]
        self.seconds += time.perf_counter()-started
        return result

    def stats(self):
        return {'model': self.manifest['model'], 'revision': self.manifest['revision'],
            'device': 'cpu', 'load_seconds': self.load_seconds, 'scoring_seconds': self.seconds,
            'scoring_calls': self.calls, 'texts_encoded': self.texts_encoded,
            'encoding_windows': self.encoding_windows, 'long_inputs_encoded_in_full': self.long_inputs,
            'vector_cache_hits': self.cache_hits, 'generative_model_calls': 0,
            'purpose': 'Rank literal source windows; scores are not entailment probabilities.'}


class SemanticSources:
    """Question-independent vectors over the existing, unchanged source leaves."""
    def __init__(self, index, similarity, directory='artifacts/semantic_source_vectors'):
        started = time.perf_counter()
        self.index, self.similarity = index, similarity
        if hashlib.sha256(index.context.encode('utf-8')).hexdigest() != index.context_hash:
            raise ValueError('Semantic cache original-context hash differs')
        spans = [list(span) for span in index.spans]
        if len(spans) != len(index.texts) or any(index.context[a:b] != text for (a, b), text in zip(spans, index.texts)):
            raise ValueError('Semantic cache requires unchanged original source spans')
        identity = {'version': 'original-leaf-vectors-v1', 'context_sha256': index.context_hash,
            'spans_sha256': hashlib.sha256(json.dumps(spans, separators=(',', ':')).encode()).hexdigest(),
            'encoder_model': similarity.manifest['model'], 'encoder_revision': similarity.manifest['revision'],
            'encoder_files_sha256': hashlib.sha256(json.dumps(similarity.manifest.get('files', []), sort_keys=True).encode()).hexdigest(),
            'encoding': 'passage-prefix_all-overlapping-token-windows_normalized-mean'}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = Path(directory)/(key+'.npz')
        self.reused = path.exists()
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if json.loads(str(data['identity'])) != identity:
                    raise ValueError('Semantic source-cache identity changed')
                self.vectors = data['vectors']
                if data['spans'].tolist() != spans:
                    raise ValueError('Semantic source-cache span mapping changed')
        else:
            # No task, query, options, response or score enters construction.
            self.vectors = similarity.vectors(index.texts, 'passage: ')
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=key+'.', suffix='.tmp', delete=False) as stream:
                temp = Path(stream.name)
                np.savez_compressed(stream, identity=json.dumps(identity, sort_keys=True),
                                    vectors=self.vectors, spans=np.asarray(spans))
            try:
                # Create the canonical name atomically, without replacing a
                # cache another worker may already have committed.
                os.link(temp, path)
            except FileExistsError:
                pass
            finally:
                temp.unlink()
            with np.load(path, allow_pickle=False) as stored:
                if json.loads(str(stored['identity'])) != identity or stored['spans'].tolist() != spans:
                    raise ValueError('Concurrent semantic source-cache identity changed')
                self.vectors = stored['vectors']
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(spans) or not np.isfinite(self.vectors).all():
            raise ValueError('Invalid semantic source-cache shape or values')
        self.metadata = {'path': path.as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'identity': identity, 'reused': self.reused, 'wall_seconds': time.perf_counter()-started,
            'source_count': len(spans), 'source_text_or_partitions_changed': False,
            'query_calls': 0, 'query_seconds': 0.0}

    def search(self, query, top_k, scope=None, *, exclude_passages=()):
        started = time.perf_counter()
        node = self.index.nodes[self.index._scope(scope)]
        vector = self.similarity.vectors([query], 'query: ')[0]
        scores = self.vectors[node['lo']:node['hi']] @ vector
        eligible = range(len(scores))
        if exclude_passages:
            eligible = (i for i in eligible if f'C{i+node["lo"]}' not in exclude_passages)
        ids = sorted(eligible, key=lambda i: (-float(scores[i]), i))[:top_k]
        result = [self.index.record(i+node['lo'], float(scores[i])) for i in ids]
        self.metadata['query_calls'] += 1
        self.metadata['query_seconds'] += time.perf_counter()-started
        return result
