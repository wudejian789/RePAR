"""Select complementary original-source windows for a grounded answer."""
from collections import Counter
import math

from repar.retrieval import lexical


def pack_answer(memory, task, trace, budget):
    queries = list(dict.fromkeys([task, *(
        execution["parameters"]["query"]
        for step in trace for execution in step["executed"]
        if execution["name"] == "tree_search" and execution["parameters"].get("query")
    )]))
    records = list(memory.records.values())
    if not records:
        return "", []
    frequency = Counter()
    words = {}
    for record in records:
        tokens = set(lexical.terms(record["text"]))
        words[record["id"]] = tokens
        frequency.update(tokens)
    weights = {term: math.log1p(len(records) / (count + 0.5)) for term, count in frequency.items()}
    candidates = {}
    for query in queries:
        terms = set(lexical.terms(query)) & weights.keys()
        ranked = sorted(records, key=lambda record: (
            -sum(weights[term] for term in terms & words[record["id"]]) / math.sqrt(len(words[record["id"]]) or 1),
            record["start"],
        ))
        for rank, record in enumerate(ranked):
            if not terms & words[record["id"]]:
                break
            view = memory._best(record, query, min(budget, 224))
            if view is None or not view["text"].strip():
                continue
            key = record["id"], view["start"], view["end"]
            if key not in candidates:
                candidates[key] = [view, 0.0]
            candidates[key][1] += 1 / (10 + rank + 1)
    if not candidates:
        return memory.render(task, budget, stage="answer")

    chosen, normalized = [], set()
    for view, _ in sorted(candidates.values(), key=lambda pair: (-pair[1], pair[0]["start"])):
        text = " ".join(view["text"].split())
        if text in normalized or any(
            item["id"] == view["id"] and view["start"] < item["end"] and item["start"] < view["end"]
            for item in chosen
        ):
            continue
        trial = [*chosen, view]
        if memory.count("\n\n".join(memory.block(item) for item in trial)) <= budget:
            chosen = trial
            normalized.add(text)
    if not chosen:
        return memory.render(task, budget, stage="answer")
    memory.mark_visible(chosen, "answer")
    return "\n\n".join(memory.block(item) for item in chosen), chosen
