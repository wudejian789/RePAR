"""Resume choice grading with focused recovery for judge quotations that fail validation."""
import json
from pathlib import Path

from repar.io import read_json, write_json
from repar.model_support import BoundedLLMClient

import regrade_math_2025 as grading
from resume_math_2025_regrade import choice_observation, recovery_manifest
from run_benchmark_recommended import sha


class QuoteRepairClient(BoundedLLMClient):
    def complete(self, messages, max_tokens, *, json_mode, schema, tag):
        raw = super().complete(messages, max_tokens, json_mode=json_mode, schema=schema, tag=tag)
        if tag != "math_choice_format_repair_2":
            return raw
        data = json.loads(messages[1]["content"])
        for instruction in (
            "上次 quote 无法在候选原答定位。请重读候选，只复制原文中最短的、足以支持选择的片段。不要添加标题、标点或复述。多个分散的原文片段用省略号连接。若没有明确最终选择，返回空数组和空引文。",
            "重新独立检查最终选择。quote 是从候选原答复制的子串，不能添加字符。可引用明确结论所在行的一小段，不必复制整行。仍在试算或猜测时返回空数组及空字符串。",
        ):
            try:
                choice_observation(data["question"], data["candidate_answer"], raw)
                return raw
            except (ValueError, KeyError, TypeError):
                repair = [*messages[:2], {"role": "assistant", "content": raw}, {"role": "user", "content": instruction}]
                raw = super().complete(repair, max_tokens, json_mode=json_mode, schema=schema, tag="math_choice_quote_repair")
        return raw


def verify_recovery(*, create=False):
    recovery_manifest()
    path = grading.ROOT / "quote_repair.json"
    identity = {"implementation_sha256": sha(Path(__file__)), "prior_protocol_sha256": sha(grading.ROOT / "quote_validation.json")}
    if path.exists():
        assert read_json(path) == identity
    else:
        assert create
        write_json(path, identity)


if __name__ == "__main__":
    verify_recovery(create=True)
    grading.choice_observation = choice_observation
    grading.BoundedLLMClient = QuoteRepairClient
    grading.main()
