"""Exact source spans packed with the active model's tokenizer."""
from __future__ import annotations

from functools import lru_cache
import re

from repar.text import terms


class EvidenceMemory:
    def __init__(self, context, client, token_budget=7000):
        self.context, self.client, self.budget = context, client, token_budget
        self.records, self.entities, self.events, self.tree_nodes = {}, {}, [], {"T0"}
        self.count = lru_cache(maxsize=4096)(lambda text: len(client.tokenize(text)))

    def observe(self, observation, step):
        added = []
        for r in observation.get("results", []):
            if self.context[r["start"]:r["end"]] != r["text"]:
                raise ValueError("Tool source does not match original offsets")
            if r["id"] not in self.records:
                added.append(r["id"])
            self.records[r["id"]] = r
            self.entities.update({e["id"]: e for e in r.get("entities", [])})
            self.tree_nodes.update(r.get("tree_path", []))
        self.entities.update({e["id"]: e for e in observation.get("entities", [])})
        for node in [observation.get("node"), *observation.get("children", [])]:
            if isinstance(node, dict) and "id" in node:
                node_id = node["id"]
                self.tree_nodes.add(node_id if isinstance(node_id, str) else f"T{node_id}")
        self.events.append({"step": step, "status": observation.get("status"), "new_sources": added})
        return added

    def _window(self, record, query, cap):
        text, start = record["text"], record["start"]
        if self.count(text) <= cap:
            return dict(record)
        # Select an exact contiguous source window; no rewriting/cleaning of source.
        q = set(terms(query))
        boundaries = [0, *[m.end() for m in re.finditer(r"[。！？.!?]\s*|\n\n", text)], len(text)]
        ranges = [(a, min(len(text), max(b, a+200))) for a, b in zip(boundaries, boundaries[1:]) if b > a]
        best = max(ranges, key=lambda pair: (len(q & set(terms(text[pair[0]:pair[1]]))), -pair[0])) if ranges else (0, len(text))
        lo = max(0, best[0]-180)
        hi = min(len(text), max(best[1]+500, lo+1400))
        while hi > lo and self.count(text[lo:hi]) > cap:
            hi = lo + (hi-lo)*3//4
        return {**record, "start": start+lo, "end": start+hi, "text": text[lo:hi], "parent_source_id": record["id"]}

    def render(self, query, budget=None):
        budget = self.budget if budget is None else budget
        q = set(terms(query))
        ranked = sorted(self.records.values(), key=lambda r: (-len(q & set(terms(r["text"]))), -r.get("score", 0), r["start"]))
        chosen, blocks, used = [], [], 0
        for record in ranked:
            selected = self._window(record, query, min(850, max(80, budget-used)))
            block = f"[{selected['id']} chars={selected['start']}:{selected['end']}]\n{selected['text']}"
            cost = self.count(block)+2
            if used+cost <= budget:
                chosen.append(selected)
                blocks.append(block)
                used += cost
            if budget-used < 80:
                break
        # Actual serialization count, including boundaries, is the final authority.
        while blocks and self.count("\n\n".join(blocks)) > budget:
            blocks.pop(); chosen.pop()
        return "\n\n".join(blocks), chosen
