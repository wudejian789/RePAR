"""Evaluation adapter for the unchanged v2 workflow under another backbone.

Shares v3_suite's dataset adapters, final generation and scoring. This layer
normalizes recorded research results only; it never changes the v2 algorithm.
"""
from types import SimpleNamespace

from repar.benchmarks.v3_suite import Suite
from repar.io import append_jsonl
from repar.workflow import research


class V2Suite(Suite):
    def research_result(self, task, index, client, dataset, qid, *, longbench_item=None):
        result = research(task, index, client, self.solver)
        records = [result.memory.records[cid] for cid in result.evidence_ids]
        events = self.output/dataset/"events"/(qid+".jsonl")
        # v2 supplies its completed trace only upon return. These are exports
        # of that real trace, not claimed to be live v3-style event timestamps.
        for step in result.trace:
            if step.get("executed"):
                append_jsonl(events, {"stage": "observation", "origin": "v2_returned_trace",
                    "round": step["round"], "executed": step["executed"]})
            if "reflection" in step:
                append_jsonl(events, {"stage": "reflection", "origin": "v2_returned_trace",
                    "round": step["round"], "reflection": step["reflection"]})
        return SimpleNamespace(evidence=result.evidence, evidence_ids=result.evidence_ids,
            evidence_records=records, trace=result.trace, termination=result.termination,
            state={"policies": result.policies, "trace_export_origin": "v2_returned_trace"})
