"""Keep the current task, source text, and provisional notes distinct."""
from xml.sax.saxutils import escape


def separated_answer_input(task, evidence, note=""):
    sections = [("task", task), ("source_material", evidence)]
    if note:
        sections.append(("provisional_note", note))
    return "\n\n".join(f"<{label}>\n{escape(text)}\n</{label}>" for label, text in sections)
