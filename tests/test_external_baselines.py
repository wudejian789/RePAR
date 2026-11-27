"""Independent invented facts exercise actual upstream algorithms, not test answers."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from baselines._upstream import index_identity
from repar.model_support import BoundedLLMClient, ExecutionBudgetExceeded


def optional_dependencies(*names):
    for name in names:
        pytest.importorskip(name, reason='Optional upstream baseline dependency; install .[baselines]')


class FixtureEncoder:
    dimension = 32
    identity = 'independent-fixture-encoder-v1'

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for text in texts:
            vector = np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8).astype(float) + 1
            result.append(vector / np.linalg.norm(vector))
        return np.asarray(result)


class FixtureClient:
    config = {'model': 'fixture', 'temperature': .1, 'seed': 42, 'deployment_id': 'fixture'}
    base = 'http://unused.invalid'

    def __init__(self):
        self.calls = []

    def complete(self, messages, max_tokens=128, **kwargs):
        text = '\n'.join(m['content'] for m in messages)
        if kwargs.get('tag') == 'raptor_summary':
            output = 'Mira operates the Lumen observatory. The station measures distant stars.'
        elif kwargs.get('tag') == 'hipporag_infer':
            if 'fact_before_filter' in text:
                output = '[[ ## fact_after_filter ## ]]\n{"fact": [["mira", "operates", "lumen"]]}\n[[ ## completed ## ]]'
            elif 'triples' in text.lower():
                output = json.dumps({'triples': [['Mira', 'operates', 'Lumen']]})
            else:
                output = json.dumps({'named_entities': ['Mira', 'Lumen']})
        elif 'high_level_keywords' in text:
            output = json.dumps({'high_level_keywords': ['observatory'], 'low_level_keywords': ['Mira', 'Lumen']})
        elif 'YES' in text and 'NO' in text:
            output = 'NO'
        else:
            output = ('("hyper-relation"<|>"Mira operates Lumen observatory"<|>9)##'
                      '("entity"<|>"Mira"<|>"person"<|>"Mira operates Lumen"<|>95)##'
                      '("entity"<|>"Lumen"<|>"organization"<|>"Lumen is an observatory"<|>90)<|COMPLETE|>')
        self.calls.append({'cached': False, 'response': {'usage': {}, 'choices': [{'finish_reason': 'stop'}]}})
        return output


def test_grammar_and_deadline(tmp_path):
    import time
    config = {**FixtureClient.config, 'base_url': FixtureClient.base, 'enable_thinking': False, 'answer_grammar': 'choice'}
    client = BoundedLLMClient(config, tmp_path, deadline=time.monotonic()-1)
    assert 'answer_grammar' not in client.config
    with pytest.raises(ExecutionBudgetExceeded):
        client.complete([{'role': 'user', 'content': 'free task'}])


def test_index_identity_refuses_changed_context(tmp_path):
    args = ('HippoRAG2', 'source one', FixtureClient(), FixtureEncoder(), {'chunk_words': 20}, tmp_path)
    assert index_identity(*args) == index_identity(*args)
    with pytest.raises(ValueError, match='Incompatible'):
        index_identity('HippoRAG2', 'source two', args[2], args[3], args[4], tmp_path)


def test_raptor_real_clustering_and_summary_provenance():
    optional_dependencies('torch', 'umap', 'sentence_transformers', 'openai', 'tiktoken')
    from baselines.RaptorRAG import RaptorRetriever
    context = ' '.join(f'Mira observes star {i} with instrument {i*i}.' for i in range(32))
    client = FixtureClient()
    source = RaptorRetriever(context, client, FixtureEncoder(), chunk_words=8, num_layers=1)
    assert source.stats['summaries'] > 0
    assert client.calls
    records = source.retrieve('Mira and Lumen', 4)
    assert records
    for record in records:
        assert record['source_spans']
        if record['kind'] == 'source':
            span = record['source_spans'][0]
            assert context[span['start']:span['end']] == record['text']


def test_hippo_openie_ppr_and_sources(tmp_path):
    optional_dependencies('torch', 'igraph', 'openai', 'pandas', 'pyarrow', 'litellm')
    from baselines.HippoRAG2 import HippoRetriever
    context = 'Mira operates Lumen observatory. Lumen records the brightness of distant stars.'
    source = HippoRetriever(context, FixtureClient(), FixtureEncoder(), working_dir=tmp_path, chunk_words=8)
    try:
        records = source.retrieve('Who operates Lumen?', 2)
        assert source.stats['graph_edges'] > 0
        assert source.stats['graph_seeds']
        assert records
        assert all(context[r['start']:r['end']] == r['text'] for r in records)
    finally:
        source.close()


def test_hypergraph_nary_evidence(tmp_path):
    optional_dependencies('torch', 'aioboto3', 'ollama', 'nano_vectordb', 'tiktoken')
    from baselines.HyperGraphRAG import HyperGraphRetriever
    source = HyperGraphRetriever('Mira operates Lumen observatory.', FixtureClient(), FixtureEncoder(), working_dir=tmp_path)
    assert source.stats['hyperedges'] > 0
    records = source.retrieve('Who operates Lumen?', 2)
    assert records and records[0]['kind'] == 'upstream_context_bundle'
    assert Path(records[0]['provenance_store']).is_file()


def test_dataset_prompt_and_delegates_are_identical(monkeypatch, tmp_path):
    from repar.evaluation import official_prompt
    from baselines.evaluation.longbench import prompt, answer
    from baselines.RaptorRAG.adapters.longbench import answer as raptor
    from baselines.HippoRAG2.adapters.longbench import answer as hippo
    from baselines.HyperGraphRAG.adapters.longbench import answer as hyper
    item = {'question': 'Which observatory?', **{'choice_'+x: x+' station' for x in 'ABCD'}}
    expected = official_prompt(item, 'independent evidence', rag=True)
    monkeypatch.chdir(tmp_path)
    assert prompt(item, 'independent evidence') == expected
    assert raptor is hippo is hyper is answer


def test_factory_control_matches_existing_bm25():
    from baselines._factory import create_retriever
    from baselines.BM25_RAG import BM25Retriever
    from baselines.BM25_RAG.pipeline import retrieve_context
    text = 'Mira operates Lumen. Orin maintains Cedar station.'
    old = BM25Retriever(text, 4)
    new = create_retriever('BM25_RAG', text, chunk_words=4)
    assert new.retrieve('Lumen', 2) == retrieve_context('Lumen', old, 2)[1]
