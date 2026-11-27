"""Full baseline evaluation using existing Direct/BM25 prompts and operators.

No research workflow or tree/graph index is involved. Dataset formatting and
the original truncation policies remain here, outside the baseline algorithms.
"""
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

from baselines.BM25_RAG import BM25Retriever, retrieve_context
from repar.benchmarks.math_exam import task_text, solver_messages
from repar.benchmarks.math_suite import count_prompt_tokens
from repar.benchmarks.v3_suite import Suite, timestamp
from repar.evaluation import direct, official_prompt
from repar.io import read_json, read_jsonl, write_json, digest


class BaselineSuite(Suite):
    def load_index(self, context, kind):
        if self.config["variant"] == "baseline_bm25":
            index = BM25Retriever(context, self.solver["chunk_words"])
            return index, {"kind": "bm25_only", "context_sha256": digest(context),
                           "stats": index.stats, "tree_or_graph_built": False}
        return SimpleNamespace(context=context, context_hash=digest(context)), {
            "kind": "direct_context", "context_sha256": digest(context), "tree_or_graph_built": False}

    def math_direct_context(self, context, client):
        path, meta_path = self.output/"direct_context.txt", self.output/"direct_context.json"
        if path.exists():
            text, meta = path.read_bytes().decode("utf-8"), read_json(meta_path)
            if digest(text) != meta["sha256"]:
                raise ValueError("Corrupt selected direct context")
            return text, meta
        questions = read_jsonl(self.config["math_questions"])
        reserve = max(count_prompt_tokens(client, solver_messages(task_text(q), "")) for q in questions)+256
        budget = self.config["math_direct_input_budget"]-reserve
        if budget <= 0:
            raise ValueError("No input budget remains after preserving the complete question")
        tokens = client.tokenize(context)
        truncated = len(tokens) > budget
        selected = tokens[:budget//2]+tokens[-(budget-budget//2):] if truncated else tokens
        text = client.detokenize(selected) if truncated else context
        meta = {"original_context_tokens": len(tokens), "retained_context_tokens": len(selected),
            "context_token_budget": budget, "task_and_template_reserve": reserve,
            "input_budget": self.config["math_direct_input_budget"], "truncated": truncated,
            "policy": "fixed head/tail context, full question preserved, same subset for all questions",
            "sha256": digest(text)}
        # Save selected artifacts without Windows newline translation. The
        # original corpus remains untouched and retains all original noise.
        path.write_bytes(text.encode("utf-8")); write_json(meta_path, meta)
        return text, meta

    def generate(self, qid, task, index, dataset, longbench_item=None):
        client = self.client()
        started = time.perf_counter()
        method = "direct" if self.config["variant"] == "baseline_direct" else "bm25"
        row = {"id": qid, "time": timestamp(), "status": "ok", "task": task, "response": "",
            "context_sha256": index.context_hash, "agent_version": self.solver["agent_version"],
            "method": method, "trace": [], "evidence_ids": [], "evidence_records": [],
            "termination": "single_final_generation"}
        try:
            if method == "direct" and dataset == "longbench":
                config = {**self.solver, "answer_max_tokens": 128,
                          "direct_input_budget": self.config["longbench_direct_input_budget"]}
                client.config["answer_grammar"] = self.config["longbench_answer_grammar"]
                row.update(direct(longbench_item, client, config))
                final = client.calls[-1]
                row["rendered_input_tokens"] = final["response"]["usage"]["prompt_tokens"]
            else:
                if method == "direct":
                    evidence, meta = self.math_direct_context(index.context, client)
                    row["direct"] = meta
                else:
                    evidence, records = retrieve_context(task, index, self.solver["retrieval_top_k"])
                    row.update({"evidence_ids": [r["id"] for r in records], "evidence_records": records,
                                "index_stats": index.stats})
                if dataset == "math":
                    messages, maximum, tag = solver_messages(task, evidence or "未取得相关教材内容。"), 4096, "math_"+method+"_answer"
                else:
                    messages = [{"role": "user", "content": official_prompt(longbench_item, evidence, rag=True)}]
                    maximum, tag = 128, "bm25_answer"
                    client.config["answer_grammar"] = self.config["longbench_answer_grammar"]
                row["rendered_input_tokens"] = count_prompt_tokens(client, messages)
                limit = self.config["math_direct_input_budget"] if dataset == "math" else self.solver["context_window"]-maximum-128
                if row["rendered_input_tokens"] > limit:
                    raise ValueError("Input exceeds frozen budget; no additional evidence truncation")
                row["response"] = client.complete(messages, maximum, tag=tag)
            if not row["response"].strip():
                raise ValueError("Empty final response")
        except Exception as exc:
            row.update({"status": "error", "error": str(exc), "traceback": traceback.format_exc()})
        row.update({"seconds": time.perf_counter()-started, "calls": client.calls})
        write_json(self.output/dataset/"traces"/(qid+".json"), row)
        return row
