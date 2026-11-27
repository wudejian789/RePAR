"""Optional retrieval encoder boundaries, cache isolation, and raw evidence."""
from collections import OrderedDict
import json

import numpy as np
import pytest

from repar.v4.semantic import SourceSimilarity, load_encoder
from repar.v4.memory import EvidenceMemory


class Tokenizer:
    def encode(self, text, **kw): return [ord(c) for c in text]
    def decode(self, ids, **kw): return ''.join(chr(c) for c in ids)


class Encoder:
    max_seq_length = 24
    tokenizer = Tokenizer()
    def __init__(self): self.inputs = []
    def encode(self, values, **kw):
        self.inputs.extend(values)
        vectors = np.asarray([[1.0, float('custody' in text or 'keeps' in text)] for text in values])
        return vectors/np.linalg.norm(vectors, axis=1, keepdims=True)


def scorer(limit=8):
    value = SourceSimilarity.__new__(SourceSimilarity)
    value.model = Encoder()
    value.cache, value.cache_limit, value.batch_size = OrderedDict(), limit, 8
    value.calls = value.cache_hits = value.texts_encoded = value.encoding_windows = value.long_inputs = 0
    value.seconds = value.load_seconds = 0
    value.manifest = {'model': 'independent-fixture', 'revision': 'fixture'}
    return value


def test_long_input_covers_every_token_and_keeps_suffix():
    value = scorer()
    text = 'abcde '*20+'terminal_marker'
    pieces = value.pieces(text, 'passage: ')
    assert len(pieces) > 1 and pieces[-1].endswith('terminal_marker')
    ids = value.model.tokenizer.encode(text)
    capacity = value.model.max_seq_length-8
    stride = capacity*3//4
    covered = {i for start in range(0, len(ids), stride) for i in range(start, min(start+capacity, len(ids)))}
    assert covered == set(range(len(ids)))
    assert all(len(piece.removeprefix('passage: ')) <= capacity for piece in pieces)


def test_encoder_prefix_cache_and_eviction_do_not_drop_batch_results():
    value = scorer(limit=2)
    first = value.scores('custody', ['keeps', 'catalog'])
    assert len(first) == 2 and first[0] > first[1]
    assert any(s.startswith('query: ') for s in value.model.inputs)
    assert any(s.startswith('passage: ') for s in value.model.inputs)
    assert len(value.cache) <= 2
    result = value.vectors(['one', 'two', 'three', 'four'], 'passage: ')
    assert result.shape == (4, 2) and len(value.cache) == 2
    other = scorer()
    assert not other.cache


def test_frozen_manifest_identity_is_checked_before_model_load(tmp_path):
    (tmp_path/'download_manifest.json').write_text(json.dumps({'files': []}), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest differs'):
        load_encoder(str(tmp_path), 'invalid expected hash', 2)


def test_model_file_hash_is_checked_before_optional_library_load(tmp_path):
    import hashlib
    (tmp_path/'weights').write_bytes(b'changed')
    raw = json.dumps({'files': [{'path': 'weights', 'sha256': 'wrong'}]}).encode()
    (tmp_path/'download_manifest.json').write_bytes(raw)
    with pytest.raises(ValueError, match='encoder file changed'):
        load_encoder(str(tmp_path), hashlib.sha256(raw).hexdigest(), 2)


def test_semantic_window_rank_preserves_original_source_and_offsets():
    class CharacterClient:
        def tokenize(self, text): return list(text)
    class Meaning:
        def scores(self, query, texts):
            return [1.0 if 'Elm Library keeps the files' in text else 0.0 for text in texts]
    common = 'Archive custody transfer register procedures are recorded. '
    fact = 'Elm Library keeps the files. '
    text = common*8+'\n\n'+fact+'Only supervised access is permitted. '
    memory = EvidenceMemory(text, CharacterClient(), 130, Meaning())
    record = {'id': 'C0', 'start': 0, 'end': len(text), 'text': text}
    memory.observe({'results': [record]})
    _, views = memory.discover('Archive custody transfer register procedures', 130, stage='observation')
    assert views and fact.strip() in views[0]['text']
    assert all(text[r['start']:r['end']] == r['text'] for r in views)
    evidence, final = memory.render('Who holds the files?', 130)
    assert final and fact.strip() in evidence and len(evidence) <= 130
