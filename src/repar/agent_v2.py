"""LongBench adapter. Benchmark formatting never enters the general controller."""
from pathlib import Path

from .evaluation import extract_answer,official_prompt,qa_text
from .io import write_json
from .workflow import research


def answer(item,index,client,config,method="repar"):
    if method == "bm25":
        from baselines.BM25_RAG.adapters.longbench import answer as baseline
        return baseline(item,index,client,config)
    result=research(qa_text(item),index,client,config,allow_graph=method=="repar")
    prompt=official_prompt(item,result.evidence or "No source evidence was obtained.",rag=True)
    response=client.complete([{"role":"user","content":prompt}],config["answer_max_tokens"],tag=f"v2_{method}_answer")
    path=Path(config["trace_dir"])/method/(item["_id"]+".json")
    write_json(path,{"_id":item["_id"],"question":qa_text(item),"context_sha256":index.context_hash,
                     "protocol":"plan_action_observe_reflect_v2","trace":result.trace,"evidence_ids":result.evidence_ids,
                     "response":response,"policies":result.policies,"termination":result.termination,"fallback":False})
    return {"response":response,"pred":extract_answer(response),"evidence_ids":result.evidence_ids,
            "trace_path":str(path),"rounds":len(result.trace),
            "tool_calls":sum(len(t.get("executed",[])) for t in result.trace),
            "termination":result.termination,"fallback":False,"index_stats":index.stats,
            "context_sha256":index.context_hash,"agent_version":"react-v2"}
