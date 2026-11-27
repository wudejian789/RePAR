"""Thin v4 adapter: unchanged data, index loading, final prompts and grading."""
import json
from functools import lru_cache

from repar.benchmarks.math_exam import formatted_solver_messages, render_problem
from repar.benchmarks.v3_suite import Suite
from repar.dense_index import DenseKnowledgeIndex, DENSE_VERSIONS, FOCUSED_DENSE_VERSIONS, V5
from repar.evaluation import official_prompt, qa_text
from repar.io import digest, read_jsonl
from repar.v4 import research
from repar.v4.prior import revise_messages
from repar.v4.working_notes import working_note
from repar.v4.workflow import EXCLUSIVE_REFLECTION_VERSIONS, FOCUSED_ACTION_VERSIONS, TASK_TEXT_PLAN_VERSIONS


ORIGINAL_INSTRUCTIONS_VERSIONS = {
    "react-v4-r48", "react-v4-r49", "react-v4-r50", "react-v4-r51", "react-v4-r52", "react-v4-r53", "react-v4-r54",
    "react-v4-r55", "react-v4-r56", "react-v4-r57",
}
READABLE_FIGURE_VERSIONS = {"react-v4-r44", "react-v4-r45", "react-v4-r46", "react-v4-r47"} | ORIGINAL_INSTRUCTIONS_VERSIONS


@lru_cache(maxsize=2)
def textbook_availability(path):
    parts, unavailable, start = [], [], 0
    for row in read_jsonl(path):
        text = row["content"]
        end = start + len(text)
        if end > start and row.get("usable_for_retrieval") is False:
            unavailable.append((start, end))
        parts.append(text)
        start = end + 2
    return digest("\n\n".join(parts)), tuple(unavailable)


class V4Suite(Suite):
    def load_index(self, context, kind):
        if self.solver["agent_version"] not in DENSE_VERSIONS:
            return super().load_index(context, kind)
        directory = self.config["dense_index_directory"]
        index = DenseKnowledgeIndex.cached(context, self.solver, directory, build=False)
        path = index.cache_path(context, self.solver, directory)
        return index, {"path": path.as_posix(), "context_sha256": index.context_hash,
                       "reused": True, "stats": index.stats}

    def answer_messages(self, task, result, longbench_item):
        version = self.solver["agent_version"]
        if version == V5:
            return revise_messages(super().answer_messages(task, result, longbench_item), result.state["direct_answer"])
        if version not in {"react-v4-r34", "react-v4-r35", "react-v4-r36"} | EXCLUSIVE_REFLECTION_VERSIONS:
            return super().answer_messages(task, result, longbench_item)
        note = working_note(result)
        if longbench_item is None:
            return formatted_solver_messages(task, result.evidence or "未取得相关教材内容。", note,
                                             preserve_premises=version in EXCLUSIVE_REFLECTION_VERSIONS,
                                             linear_notation=version == "react-v4-r38", task_last=version == "react-v4-r39",
                                             deliver_results_as_found=version == "react-v4-r42",
                                             readable_figures=version in READABLE_FIGURE_VERSIONS,
                                             separate_sources=version == "react-v4-r46",
                                             verify_given_conditions=version == "react-v4-r47",
                                             original_instructions=version in ORIGINAL_INSTRUCTIONS_VERSIONS)
        evidence = result.evidence or "No source evidence was obtained."
        if note:
            evidence += "\n\nProvisional research note. Verify against the original passages above.\n" + note
        return [{"role": "user", "content": official_prompt(longbench_item, evidence, rag=True)}]

    def research_result(self, task, index, client, dataset, qid, *, longbench_item=None):
        version = self.solver["agent_version"]
        query_task = None
        query_anchor = None
        unavailable = ()
        if dataset == "math" and self.solver.get("respect_source_availability"):
            context_hash, unavailable = textbook_availability(self.config["math_textbook"])
            if context_hash != index.context_hash:
                raise ValueError("Availability metadata does not match the indexed textbook")
        if version in FOCUSED_DENSE_VERSIONS:
            if longbench_item is None:
                question = json.loads(task)
                query_task = render_problem(question, readable_figures=True)
                query_anchor = "\n".join([question["question_stem"], *(item["question"] for item in question["subquestions"])])
            else:
                query_anchor = longbench_item["question"]
                query_task = (qa_text(longbench_item) if version == V5 else
                              "\n".join([query_anchor, *(longbench_item["choice_" + key] for key in "ABCD")]))
        elif dataset == 'math' and version in TASK_TEXT_PLAN_VERSIONS | FOCUSED_ACTION_VERSIONS:
            question = json.loads(task)
            query_task = '\n'.join([question['question_stem'], *(item['text'] for item in question['choices']),
                                     *(item['question'] for item in question['subquestions'])])
            if self.solver.get("question_only_anchor"):
                query_anchor = "\n".join([question["question_stem"], *(item["question"] for item in question["subquestions"])])
            if version == "react-v4-r52":
                query_anchor = query_anchor or query_task
                task = render_problem(question, readable_figures=True)
                query_task = task
        elif self.solver.get("question_only_anchor") and longbench_item is not None:
            query_anchor = longbench_item["question"]
        if version in FOCUSED_ACTION_VERSIONS and longbench_item is not None:
            query_task = '\n'.join([longbench_item['question'], *(longbench_item['choice_'+key] for key in 'ABCD')])
        return research(task, index, client, self.solver,
                        event_path=self.output/dataset/"events"/(qid+".jsonl"), query_task=query_task,
                        unavailable_ranges=unavailable, query_anchor=query_anchor)
