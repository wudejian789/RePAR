"""Carry a source-linked observation into the answer as a provisional note."""


def working_note(result):
    visible = set(result.evidence_ids)
    final_windows = {}
    for view in result.evidence_records:
        final_windows.setdefault(view["id"], []).append((view["start"], view["end"]))
    for step in reversed(result.trace):
        reflection = step.get("reflection", {})
        observed = set(step.get("reflection_context", {}).get("available_sources", []))
        citations = list(dict.fromkeys(reflection.get("supported_sources", [])))
        assessment = reflection.get("assessment", "").strip()
        spans = reflection.get("source_spans", [])
        if spans and not all(
            any(start <= span["start"] and span["end"] <= end for start, end in final_windows.get(span["id"], ()))
            for span in spans
        ):
            continue
        if assessment and citations and set(citations) <= (visible & observed):
            note = f"[{', '.join(citations)}] {assessment}"
            if spans and (lesson := reflection.get("lesson", "").strip()):
                note += "\n" + lesson
            return note
    return ""
