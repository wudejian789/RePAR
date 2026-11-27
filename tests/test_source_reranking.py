"""Coverage and identity of generic source reranking windows."""
import pytest

from repar.v4.rerank import token_windows


@pytest.mark.parametrize('size,capacity', [(0, 20), (4, 20), (20, 20), (21, 20), (45, 20), (83, 20), (7, 1)])
def test_pair_windows_cover_all_tokens_and_respect_capacity(size, capacity):
    tokens = list(range(size))
    pieces = token_windows(tokens, capacity)
    assert pieces and all(len(p) <= capacity for p in pieces)
    assert set(i for p in pieces for i in p) == set(tokens)
    assert pieces[-1] == tokens[-capacity:]


def test_pair_windows_reject_invalid_capacity():
    with pytest.raises(ValueError, match='capacity'):
        token_windows([1, 2], 0)


@pytest.mark.parametrize('batch_by_length', [False, True])
def test_long_query_and_source_suffix_both_reach_pair_model(batch_by_length):
    from types import SimpleNamespace
    import torch
    from repar.v4.rerank import SourceReranker

    class Tokenizer:
        model_input_names = ['input_ids', 'attention_mask']
        def encode(self, text, **kwargs): return [int(s) for s in text.split()]
        def num_special_tokens_to_add(self, **kwargs): return 4
        def prepare_for_model(self, query, pair_ids, **kwargs):
            ids = [-1, *query, -2, *pair_ids, -3]
            return {'input_ids': ids, 'attention_mask': [1]*len(ids)}
        def pad(self, rows, **kwargs):
            width = max(len(r['input_ids']) for r in rows)
            return {'input_ids': torch.tensor([r['input_ids']+[0]*(width-len(r['input_ids'])) for r in rows])}
    class Model:
        def __init__(self): self.seen = []
        def __call__(self, input_ids):
            self.seen.extend(input_ids.tolist())
            return SimpleNamespace(logits=input_ids.max(dim=1).values.float().reshape(-1, 1))
    ranker = SourceReranker.__new__(SourceReranker)
    ranker.tokenizer, ranker.model = Tokenizer(), Model()
    ranker.max_length, ranker.batch_size = 32, 3
    ranker.batch_by_length = batch_by_length
    ranker.long_queries = ranker.long_sources = ranker.windows = ranker.pairs = 0
    query, source = list(range(1, 42)), list(range(1000, 1099))
    result = ranker.score_pairs([(' '.join(map(str, query)), ' '.join(map(str, source))), ('1 2', '3 4')])
    assert result == [1098.0, 4.0]
    assert ranker.long_queries == 1 and ranker.long_sources > 0 and ranker.pairs == 2
    seen = {i for row in ranker.model.seen for i in row if i > 0}
    assert seen == set(query+source)
    assert all(len(row) <= 32 for row in ranker.model.seen)
