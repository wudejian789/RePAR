from repar.v4.answer_evidence import pack_answer
from repar.v4.memory import EvidenceMemory
from test_v4 import TokenClient


def test_answer_evidence_combines_focus_queries_and_preserves_original_ranges():
    pieces = [
        "The Fern deposit is filed under ledger A17.",
        "The recipient for A17 was the Moorland Archive.",
        "A baker opened a village shop.",
        "The Fern deposit is filed under ledger A17.",
    ]
    context = "\n\n".join(pieces)
    memory = EvidenceMemory(context, TokenClient(), source_view_policy="query_windows")
    records, offset = [], 0
    for i, piece in enumerate(pieces):
        records.append({"id": f"C{i}", "start": offset, "end": offset + len(piece), "text": piece})
        offset += len(piece) + 2
    memory.observe({"results": records})
    trace = [{"executed": [{"name": "tree_search", "parameters": {"query": "ledger A17 recipient"}}]}]
    text, views = pack_answer(memory, "Who received the Fern deposit?", trace, 180)
    assert {r["id"] for r in views} == {"C0", "C1"}
    assert text.count(pieces[0]) == 1
    assert all(r["text"] == context[r["start"]:r["end"]] for r in views)
    assert memory.exposures[-1]["stage"] == "answer"
    assert len(text) <= 180
