"""Independent checks that the backbone adapter preserves method boundaries."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from repar.benchmarks import backbone_suite, v3_suite
from repar.benchmarks.math_exam import solver_messages
from repar.evaluation import official_prompt

ROOT = Path(__file__).resolve().parents[1]


def test_27b_configs_change_model_connection_not_controller_or_index():
    allowed = {"base_url", "model", "deployment_id", "context_window", "direct_input_budget",
               "answer_max_tokens", "answer_grammar", "trace_dir"}
    for version in ("v2", "v3"):
        before = json.loads((ROOT/f"configs/repar_{version}.json").read_text(encoding="utf-8"))
        after = json.loads((ROOT/f"configs/repar_{version}_27b.json").read_text(encoding="utf-8"))
        for key in set(before) | set(after):
            if key not in allowed: assert before.get(key) == after.get(key), (version, key)
        assert after["base_url"] == "http://127.0.0.1:9090"
        assert after["context_window"] == 131072 and after["answer_max_tokens"] == 4096
        assert after["enable_thinking"] is False
        assert after["temperature"] == .1 and after["seed"] == 42
        assert "answer_grammar" not in after


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("dataset", ["math", "longbench"])
def test_same_raw_final_generation_for_both_versions(tmp_path, monkeypatch, version, dataset):
    record = {"id": "C0", "start": 0, "end": 28, "text": "Aster Archive opened in 2018."}
    evidence = "[C0] Aster Archive opened in 2018."
    def research(task, index, client, config, **kwargs):
        assert task == "original task"
        if version == "v2":
            assert not kwargs
            return SimpleNamespace(evidence=evidence, evidence_ids=["C0"],
                memory=SimpleNamespace(records={"C0": record}), trace=[], policies=[],
                termination="model_sufficient")
        assert "event_path" in kwargs
        return SimpleNamespace(evidence=evidence, evidence_ids=["C0"], evidence_records=[record],
            trace=[], state={}, termination="model_sufficient")
    monkeypatch.setattr(backbone_suite if version == "v2" else v3_suite, "research", research)
    cls = backbone_suite.V2Suite if version == "v2" else v3_suite.Suite
    suite = cls.__new__(cls); suite.output = tmp_path
    suite.solver = {"agent_version": "react-v2" if version == "v2" else "react-v3-core-r1",
                   "answer_max_tokens": 4096, "context_window": 131072}
    suite.config = {"longbench_final_max_tokens": 128, "longbench_answer_grammar": "original-grammar"}
    class Client:
        def __init__(self): self.config = {}; self.calls = []
        def tokenize(self, text): return list(text)
        def post(self, route, body): return {"prompt": "rendered"}
        def complete(self, messages, budget, **kwargs):
            self.calls.append((messages, budget, kwargs)); return 'unchanged {"unfinished"'
    client = Client();suite.client = lambda:client
    item = {"question": "When did the archive open?", **{f"choice_{letter}":letter for letter in "ABCD"}}
    row = suite.generate("independent-fixture", "original task", SimpleNamespace(context_hash="fixture"),
        dataset, longbench_item=item if dataset == "longbench" else None)
    assert row["status"] == "ok" and row["response"] == 'unchanged {"unfinished"'
    assert len(client.calls) == 1 and row["evidence_records"] == [record]
    messages, budget, kwargs = client.calls[0]
    expected = solver_messages("original task", evidence) if dataset == "math" else [
        {"role": "user", "content": official_prompt(item, evidence, rag=True)}]
    assert messages == expected and budget == (4096 if dataset == "math" else 128)
    assert not any(key in row for key in ("review", "draft", "answer", "output_checks"))
