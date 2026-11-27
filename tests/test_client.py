from repar.client import LLMClient


def test_same_answer_grammar_and_separate_controller_decoding(tmp_path):
    config = {"base_url": "http://unused", "model": "synthetic", "temperature": .1,
              "seed": 42, "enable_thinking": False,
              "answer_grammar": 'root ::= "The correct answer is (" [ABCD] ")"'}
    client = LLMClient(config, cache_dir=tmp_path)
    requests = []

    def fake_post(route, body):
        requests.append(body)
        return {"choices": [{"message": {"content": "The correct answer is (A)"}}]}

    client.post = fake_post
    for tag in ("direct", "bm25_answer", "repar_answer", "repar_tree_answer"):
        client.complete([{"role": "user", "content": tag}], tag=tag)
    client.complete([{"role": "user", "content": "controller"}], json_mode=True, tag="controller")
    assert all(r["grammar"] == config["answer_grammar"] for r in requests[:4])
    assert "grammar" not in requests[4]
    assert requests[4]["response_format"] == {"type": "json_object"}


def test_deployment_identity_separates_mtp_cache(tmp_path):
    config = {"base_url": "http://unused", "model": "same-weights", "temperature": .1,
              "seed": 42, "enable_thinking": False, "deployment_id": "mtp-off"}
    client = LLMClient(config, cache_dir=tmp_path)
    calls = []
    def fake_post(route, body):
        calls.append(body)
        return {"choices": [{"message": {"content": "A"}}]}
    client.post = fake_post
    messages = [{"role":"user","content":"same prompt"}]
    client.complete(messages)
    config["deployment_id"] = "mtp-on"
    client.complete(messages)
    client.complete(messages)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert client.calls[-1]["cached"]
