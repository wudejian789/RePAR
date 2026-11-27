"""Generic relevance/display alignment over the same unchanged source leaves."""
from repar.v4.memory import EvidenceMemory


class Tokens:
    def tokenize(self, text): return text.split()


class Relevance:
    def __init__(self): self.inputs = []
    def scores(self, query, texts):
        self.inputs.append((query, list(texts)))
        return [10.0 if 'Sedge office' in text else -1.0 for text in texts]


def memory():
    text = ('The Lark collection was catalogued years ago. Administrative audit instructions are filed separately. '
            'The Bluefin collection is preserved by the Sedge office. Inspection requires an appointment. '
            'Other collections have different restrictions. General catalog maintenance follows the annual calendar.')
    rank = Relevance()
    value = EvidenceMemory(text, Tokens(), 2000, rank, source_view_policy='scored_neighborhood')
    record = {'id': 'C0', 'start': 0, 'end': len(text), 'text': text}
    value.observe({'results': [record]})
    return value, rank, record


def test_generous_budget_does_not_bypass_relevance_or_force_whole_leaf():
    value, rank, original = memory()
    shown = value.observation_view({'operator': 'tree_search', 'status': 'ok', 'results': [original]},
                                   'Bluefin custodian inspection', 2000, 'observation')
    assert len(shown['source_views']) == 1
    result = shown['source_views'][0]
    assert result['text'] != original['text'] and 'Sedge office' in result['text']
    assert rank.inputs and result['text'] in rank.inputs[-1][1]
    assert value.context[result['start']:result['end']] == result['text']
    assert value.unread('C0') is not None


def test_final_render_preserves_scored_view_without_unscored_paragraph_fill():
    value, rank, original = memory()
    value.observation_view({'operator': 'tree_search', 'status': 'ok', 'results': [original]},
                           'Bluefin custodian inspection', 2000, 'observation')
    expected = list(value.source_views.values())
    _, final = value.render('Bluefin custodian inspection', 2000)
    assert [(r['start'], r['end']) for r in final] == [(r['start'], r['end']) for r in expected]
    assert 'annual calendar' not in final[0]['text']


def test_explicitly_read_context_remains_visible_under_neighborhood_policy():
    value, rank, original = memory()
    value.observation_view({'operator': 'read', 'status': 'ok', 'results': [original],
        'requested_view': {'id': 'C0', 'start': 0, 'end': original['end']}}, 'Bluefin', 2000, 'explicit_read')
    _, final = value.render('Bluefin', 2000)
    assert final[0]['text'] == original['text']


def test_independent_sources_each_keep_their_relevant_condition():
    text = ('Pine entry: inspection requires written approval. Its records stay on site. '
            'General office maintenance follows a separate timetable.\n\n'
            'Birch entry: public consultation is permitted. Its originals may not be exported. '
            'Additional catalog maintenance is recorded annually.')
    boundary = text.index('Birch')
    records = [{'id': f'C{i}', 'start': a, 'end': b, 'text': text[a:b]}
               for i, (a, b) in enumerate([(0, boundary), (boundary, len(text))])]
    class Rank:
        def scores(self, query, snippets):
            return [10.0 if ('approval' in snippet or 'may not be exported' in snippet) else -1.0 for snippet in snippets]
    value = EvidenceMemory(text, Tokens(), 500, Rank(), source_view_policy='scored_neighborhood')
    value.observe({'results': records})
    shown = value.observation_view({'operator': 'tree_search', 'status': 'ok', 'results': records},
                                   'Compare Pine and Birch access rules', 500, 'compare')
    _, final = value.render('Compare Pine and Birch access rules', 500)
    assert {r['id'] for r in shown['source_views']} == {'C0', 'C1'}
    assert any('written approval' in r['text'] for r in final)
    assert any('may not be exported' in r['text'] for r in final)
    assert all(text[r['start']:r['end']] == r['text'] for r in final)
