"""Behavior and entrypoint regression checks for the standalone BM25 baseline."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import pytest

from baselines.BM25_RAG import BM25Retriever, answer, retrieve_context
from baselines.BM25_RAG.adapters.longbench import answer as benchmark_answer
from repar.evaluation import official_prompt, qa_text
from repar.index import KnowledgeIndex, split_spans, terms
from repar.text import split_spans as shared_spans, terms as shared_terms


CONTEXT = "\n\n".join([
    "River monitoring measures water quality. Silver sensors detect pollution in drinking water. " * 4,
    "The museum catalog describes sculpture and paintings. Visitors attend exhibitions on weekdays. " * 4,
    "Harbor Council adopted Silver sensors in 2024. River monitoring supports local residents. " * 4,
])
CONFIG = {"chunk_words": 35, "tree_fanout": 4, "graph_neighbors": 3,
          "index_version": "test", "retrieval_top_k": 4, "answer_max_tokens": 128}


class Client:
    def __init__(self): self.calls = []

    def complete(self, messages, max_tokens, tag=""):
        self.calls.append((deepcopy(messages), max_tokens, tag))
        return "The correct answer is (B)"


def test_shared_ranker_preserves_scope_ties_and_source_partition():
    baseline = BM25Retriever(CONTEXT, CONFIG["chunk_words"])
    tree = KnowledgeIndex(CONTEXT, CONFIG)
    assert split_spans is shared_spans and terms is shared_terms
    assert baseline.spans == tree.spans
    assert "".join(baseline.texts) == CONTEXT
    for query in ("Silver sensors river", "river river river river", "unknownquerytoken", ""):
        assert baseline.bm25(query, 24) == tree.bm25(query, 24)
        all_scores = dict(baseline.bm25(query, len(baseline.spans)))
        subtree = next(n for n in tree.nodes if n["children"] and n["id"] != "T0")
        scoped = tree.bm25(query, 24, scope=subtree["id"])
        assert all(subtree["lo"] <= i < subtree["hi"] for i, _ in scoped)
        assert all(score == all_scores[i] for i, score in scoped)
    assert [i for i, _ in baseline.bm25("unknownquerytoken", 4)] == [0, 1, 2, 3]


def test_generic_pipeline_uses_exact_raw_evidence_without_benchmark_fields():
    source = BM25Retriever(CONTEXT, 35)
    context, records = retrieve_context("Silver sensors", source, 3)
    assert len(records) == 3
    assert [r["start"] for r in records] == sorted(r["start"] for r in records)
    assert all(r["text"] == CONTEXT[r["start"]:r["end"]] for r in records)
    assert context == "\n\n".join(f"[{r['id']}]\n{r['text']}" for r in records)
    client = Client()
    result = answer("Summarize local sensor deployment", source, client, top_k=3, max_tokens=256)
    messages, limit, tag = client.calls[0]
    assert limit == 256 and tag == "bm25_answer"
    assert "Summarize local sensor deployment" in messages[0]["content"]
    assert "Choices:" not in messages[0]["content"]
    assert len(result["evidence_ids"]) == 3
    assert not hasattr(source, "nodes") and not hasattr(source, "entity_graph")


def test_benchmark_adapter_and_legacy_delegation_preserve_model_input():
    from repar.agent import answer as legacy_answer
    from repar.agent_v2 import answer as agent_answer
    item = {"_id": "synthetic", "context": CONTEXT, "question": "Which sensor was used?",
            **{"choice_"+c: c for c in "ABCD"}}
    source = BM25Retriever(CONTEXT, 35)
    context, records = retrieve_context(qa_text(item), source, CONFIG["retrieval_top_k"])
    expected = ([{"role": "user", "content": official_prompt(item, context, rag=True)}], 128, "bm25_answer")
    for runner in (benchmark_answer,
                   lambda i, s, c, cfg: legacy_answer(i, s, c, cfg, "bm25"),
                   lambda i, s, c, cfg: agent_answer(i, s, c, cfg, "bm25")):
        client = Client()
        result = runner(item, source, client, CONFIG)
        assert client.calls == [expected]
        assert result["pred"] == "B"
        assert result["evidence_ids"] == [r["id"] for r in records]
    assert benchmark_answer(item, None, Client(), CONFIG)["evidence_ids"] == [r["id"] for r in records]


def test_cli_baseline_does_not_build_agent_indexes(tmp_path, monkeypatch):
    from repar import cli
    import repar.index_v2
    from repar.io import read_jsonl, write_json

    def forbidden(*args, **kwargs):
        raise AssertionError("A BM25-only run must not construct a tree or graph")

    monkeypatch.setattr(KnowledgeIndex, "cached", forbidden)
    monkeypatch.setattr(repar.index_v2.KnowledgeIndex, "cached", forbidden)
    sample = {"_id": "synthetic", "domain": "Multi-Document QA", "sub_domain": "synthetic",
              "difficulty": "easy", "length": "short", "context": CONTEXT, "question": "Which sensor?",
              "answer": "B", **{"choice_"+c: c for c in "ABCD"}}
    data = tmp_path/"data.json"
    config = tmp_path/"config.json"
    write_json(data, [sample])
    write_json(config, {**CONFIG, "agent_version": "react-v2"})

    class CliClient(Client):
        def __init__(self, config): super().__init__()
        def complete(self, messages, max_tokens, tag=""):
            self.calls.append({"response": {"usage": {}, "choices": [{"finish_reason": "stop"}]},
                               "tag": tag, "cached": False})
            return "The correct answer is (B)"

    monkeypatch.setattr(cli, "LLMClient", CliClient)
    args = argparse.Namespace(config=str(config), data=str(data), output=str(tmp_path/"results"),
                              method="bm25", limit=0, workers=1, wait_for_indexes=False)
    cli.run(args)
    rows = read_jsonl(tmp_path/"results/bm25.jsonl")
    assert len(rows) == 1 and rows[0]["judge"]
    assert rows[0]["index_stats"]["version"] == "bm25-rag-v1"


def test_retrieval_only_command_uses_no_model(tmp_path, monkeypatch, capsys):
    import sys
    from baselines.BM25_RAG.__main__ import main
    source = tmp_path/"knowledge.txt"
    source.write_text(CONTEXT, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["BM25_RAG", "--context", str(source), "--task", "Silver sensors",
                                     "--retrieve-only", "--top-k", "2"])
    main()
    result = json.loads(capsys.readouterr().out)
    assert len(result["evidence"]) == 1  # default240-word chunk covers this small source
    assert "response" not in result
    assert result["context_sha256"] == BM25Retriever(CONTEXT).context_hash
    with pytest.raises(ValueError, match="positive"): BM25Retriever(CONTEXT, 0)
    with pytest.raises(ValueError, match="positive"): BM25Retriever(CONTEXT).bm25("sensor", 0)
