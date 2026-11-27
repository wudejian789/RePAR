"""Same-input MTP throughput/correctness probe; full benchmark is evaluated separately."""
import argparse
import time

from repar.client import LLMClient
from repar.io import read_json, write_json

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--output", required=True)
p.add_argument("--repeats", default=3, type=int)
args = p.parse_args()
config = read_json(args.config)
config.pop("answer_grammar", None)
config.pop("controller_schema", None)
client = LLMClient(config, cache_dir="artifacts/cache/mtp_probe")
deployment_id = config["deployment_id"]
tasks = [
    {"name": "policy_plan", "prompt": "You are a research planner. A question compares a hospital's 2020 crisis response to its 2023 digital transformation. Write 12 concise, numbered research steps to locate both original policies, collect evidence, search counterexamples, compare their intent and reach a conclusion. Do not answer the historical question."},
    {"name": "structured_trajectory", "prompt": 'Create one JSON research plan for a question comparing two scientific papers. Use the keys reflection, plan, policy_update, tool_calls, finish. tool_calls must include tree_search with parameters query and top_k, and graph_search with parameters query, seeds, hops and top_k. Assume the first paper is at K12. Plan to find the other paper and test a causal relationship. Set finish to false. Explain evidence gaps in the reflection.'},
    {"name": "long_evidence_summary", "prompt": "Evidence 1: In 2020 the city funded emergency mobile clinics to maintain immediate access. Evidence 2: In 2023 it introduced a permanent patient record platform and staff training. Evidence 3: Both plans retained physical clinics. Write a careful 250-word comparison grounded only in these three statements, identifying supported changes and what cannot be inferred."},
    {"name": "grounded_control", "prompt": "A report states that Nova City's river program began in 2021, expanded to two districts in 2022, and covered five districts in 2024. A later appendix corrects the start date to 2020 while retaining the district counts. In a short paragraph, state the corrected start year, counts in 2022 and 2024, and the increase between those years. Explain which source supports each fact. Do not invent missing information."},
]
rows = []
for rep in range(args.repeats):
    for task in tasks:
        # Include a repeat identifier in deployment identity to force actual measured calls.
        client.config["deployment_id"] = deployment_id + f"-probe-repeat-{rep}"
        messages = [{"role": "user", "content": task["prompt"]}]
        started = time.perf_counter()
        text = client.complete(messages, max_tokens=320, json_mode=task["name"] == "structured_trajectory",
                               tag=task["name"])
        call = client.calls[-1]
        rows.append({"task": task["name"], "repeat": rep, "response": text,
                     "usage": call["response"].get("usage", {}),
                     "timings": call["response"].get("timings", {}),
                     "seconds": time.perf_counter()-started, "cached": call["cached"],
                     "finish_reason": call["response"]["choices"][0].get("finish_reason")})
        print(args.output, task["name"], rep, rows[-1]["timings"], flush=True)
write_json(args.output, {"config": config, "rows": rows})
