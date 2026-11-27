"""Per-call source visibility for post-action research feedback."""
from __future__ import annotations

from .memory import EvidenceMemory


def reflection_context(memory, observation, query, budget, stage, *, retain_prior_ranges=False,
                       include_quotations=False):
    """Supply current and retained raw evidence within one shared text budget.

    A source ID is citable only when its text is in this particular payload.
    This checks visibility/provenance, not whether a model's claim is entailed.
    """
    current_ids = {r["id"] for r in observation.get("results", [])}
    prior = [r for cid, r in memory.records.items() if cid not in current_ids]
    if retain_prior_ranges:
        prior.extend(memory.records[cid] for cid in dict.fromkeys(r['id'] for r in memory.source_views.values())
                     if cid in current_ids)
    # The observation keeps the majority of the existing evidence budget.
    # Entity-only and empty observations can devote it entirely to prior text.
    current_budget = budget-budget//3 if prior and current_ids else budget
    visible = memory.observation_view(observation, query, current_budget, stage)
    current_text = "\n\n".join(EvidenceMemory.block(r) for r in visible["source_views"])
    remaining = max(0, budget-memory.count(current_text)-2)
    retained = []
    if prior and remaining >= 80:
        shown = visible['source_views'] if retain_prior_ranges else ()
        _, retained = memory.render(query, remaining, records=prior, already_shown=shown)
    existing = [{k: r[k] for k in ("id", "start", "end", "text", "source_start", "source_end",
                "truncated_before", "truncated_after", "partial_sentence")} for r in retained]
    citable = {r["id"] for r in [*visible["source_views"], *existing]}
    # A graph proposition already contains source text; accept its source only
    # if that text matches a known original source record exactly.
    quotations = []
    for p in visible.get("propositions", []):
        parent = memory.records.get(p.get("passage_id"))
        text = p.get("text")
        if parent and text and text in parent["text"]:
            start = parent["start"]+parent["text"].index(text)
            quotations.append({"id": parent["id"], "start": start, "end": start+len(text), "text": text})
            citable.add(parent["id"])
    if quotations:
        memory.mark_visible(quotations, stage+"_propositions")
    count = memory.count("\n\n".join(EvidenceMemory.block(r) for r in [*visible["source_views"], *existing]))
    # Serialization boundaries can alter tokenization; remove retained views
    # first if their joined representation crosses the configured text budget.
    while existing and count > budget:
        existing.pop()
        count = memory.count("\n\n".join(EvidenceMemory.block(r) for r in [*visible["source_views"], *existing]))
        citable = {r["id"] for r in [*visible["source_views"], *existing, *quotations]}
    if existing:
        memory.mark_visible(existing, stage+"_retained")
    return {"observation": visible, "existing_sources": existing, "available_sources": sorted(citable),
            "source_view_tokens": count, "source_view_budget": budget,
            **({"source_quotations": quotations} if include_quotations else {})}
