"""A bounded incremental search can extend a query beyond earlier hits."""
from repar.v4.knowledge import KnowledgeAPI, request_identity
from repar.v4.memory import EvidenceMemory
from repar.v4.workflow import action_candidates


class Index:
    context = 'Kite catalog.\nKite inspection rules.'

    def _leaf_id(self, cid):
        return {'C0': 0, 'C1': 1}[cid]

    def record(self, leaf, score):
        start, end = [(0, 13), (14, len(self.context))][leaf]
        return {'id': f'C{leaf}', 'text': self.context[start:end],
                'start': start, 'end': end, 'score': score}

    def execute(self, name, params):
        assert 'exclude_passages' not in params
        return {'results': [self.record(0, 1)]}

    def bm25(self, query, limit, scope):
        return [(0, 1), (1, .5)][:limit]


def test_excluding_first_hit_refills_from_scoped_lexical_results():
    api = KnowledgeAPI(Index())
    params = {'query': 'Kite rules', 'top_k': 1}
    first = api.execute('tree_search', params)
    extended = api.execute('tree_search', {**params, 'exclude_passages': ['C0']})
    assert [r['id'] for r in first['results']] == ['C0']
    assert [r['id'] for r in extended['results']] == ['C1']
    assert extended['query_channels'][0]['parameters']['exclude_passages'] == ['C0']
    assert api.execute('tree_search', {**params, 'exclude_passages': ['C0', 'C1']})['status'] == 'no_match'
    assert api.execute('tree_search', params)['query_execution'][0]['cached']


def test_extended_search_is_optional_and_only_binds_retrieved_ids():
    index = Index()
    memory = EvidenceMemory(index.context, None)
    plan = {'query': 'Kite rules'}
    config = {'offer_unseen_search': True, 'retrieval_top_k': 1}
    first = action_candidates('Review the Kite register', plan, memory, set(), config, 0)
    assert all('exclude_passages' not in a['parameters'] for a in first)
    memory.observe({'results': [index.record(0, 1)]})
    ordinary = next(a for a in first if a['name'] == 'tree_search')
    completed = {request_identity(ordinary['name'], ordinary['parameters'])}
    after = action_candidates('Review the Kite register', plan, memory, completed, config, 1)
    additional = next(a for a in after if 'exclude_passages' in a['parameters'])
    assert additional['parameters']['exclude_passages'] == ['C0']
    assert {'read', 'tree_search', 'answer'} <= {a['name'] for a in after}
    completed.add(request_identity(additional['name'], additional['parameters']))
    assert not any('exclude_passages' in a['parameters'] for a in
                   action_candidates('Review the Kite register', plan, memory, completed, config, 2))


