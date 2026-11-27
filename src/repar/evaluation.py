from __future__ import annotations

import re
from pathlib import Path


def extract_answer(response):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if not match:
        match = re.search(r"The correct answer is ([A-D])", response)
    return match.group(1) if match else None


def qa_text(item):
    return item["question"] + "\nChoices:\n" + "\n".join(
        f"({c}) {item['choice_' + c]}" for c in "ABCD")


def official_prompt(item, context, rag=False):
    template = Path("third_party/prompts/0shot_rag.txt" if rag else
                    "third_party/prompts/0shot.txt").read_text(encoding="utf-8")
    replacements = {"$DOC$": context.strip(), "$Q$": item["question"].strip()}
    replacements.update({"$C_" + c + "$": item["choice_" + c].strip() for c in "ABCD"})
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def direct(item, client, config):
    prompt = official_prompt(item, item["context"])
    tokens = client.tokenize(prompt)
    original_tokens = len(tokens)
    budget = config["direct_input_budget"]
    truncated = original_tokens > budget
    if truncated:
        half = budget // 2
        prompt = client.detokenize(tokens[:half] + tokens[-half:])
    response = client.complete([{"role": "user", "content": prompt}],
                               config["answer_max_tokens"], tag="direct")
    return {"response": response, "pred": extract_answer(response),
            "original_prompt_tokens": original_tokens, "truncated": truncated,
            "input_budget": budget, "context_sha256": __import__("hashlib").sha256(
                item["context"].encode()).hexdigest()}
