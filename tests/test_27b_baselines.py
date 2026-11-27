"""Baseline adapters preserve the original algorithms without research indexes."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.BM25_RAG import BM25Retriever, retrieve_context
from repar.benchmarks.baseline_suite import BaselineSuite
from repar.benchmarks.math_exam import solver_messages, task_text
from repar.evaluation import official_prompt
from repar.io import digest


class Client:
    def __init__(self):self.config={"enable_thinking":False};self.calls=[];self.payloads=[]
    def tokenize(self,text):return list(text)
    def detokenize(self,tokens):return "".join(tokens)
    def post(self,route,body):
        assert route=="/apply-template"
        return {"prompt":"".join(m["content"] for m in body["messages"])}
    def complete(self,messages,maximum,**kwargs):
        self.payloads.append((messages,maximum,kwargs))
        response={"choices":[{"message":{"content":"The correct answer is (A)"},"finish_reason":"stop"}],
                  "usage":{"prompt_tokens":len("".join(m["content"] for m in messages)),"completion_tokens":8}}
        self.calls.append({"response":response,"tag":kwargs["tag"],"cached":False,"key":"fixture"})
        return "The correct answer is (A)"


def suite(tmp_path,method,client):
    s=BaselineSuite.__new__(BaselineSuite);s.output=tmp_path;s.client=lambda:client
    s.solver={"agent_version":"direct" if method=="direct" else "bm25-rag-v1","chunk_words":20,
              "retrieval_top_k":2,"context_window":10000,"enable_thinking":False}
    s.config={"variant":"baseline_"+method,"math_direct_input_budget":4096,
              "longbench_direct_input_budget":1000,"longbench_answer_grammar":"original-grammar"}
    return s


def test_direct_context_uses_one_fixed_head_tail_and_retains_unicode_separators(tmp_path):
    client=Client();s=suite(tmp_path,"direct",client)
    item={"question_type":"fill_in_blank","question_stem":"When was the archive opened?",
          "choices":[],"subquestions":[],"tables":[],"figures":[]}
    qa=tmp_path/"questions.jsonl";qa.write_text(json.dumps(item)+"\n",encoding="utf-8")
    s.config["math_questions"]=str(qa)
    context="alpha\u2028"+("x"*6000)+"omega"
    text,meta=s.math_direct_context(context,client)
    budget=meta["context_token_budget"]
    assert meta["truncated"] and text==context[:budget//2]+context[-(budget-budget//2):]
    assert text.startswith("alpha\u2028") and text.endswith("omega")
    assert meta["sha256"]==digest(text) and s.math_direct_context(context,client)==(text,meta)
    assert meta["input_budget"]==4096


@pytest.mark.parametrize("dataset",["math","longbench"])
def test_bm25_uses_existing_retrieval_and_exact_original_prompts(tmp_path,dataset):
    client=Client();s=suite(tmp_path,"bm25",client)
    context="Aster Archive opened in 2018.\n\nThe unrelated harbor closes on Sundays."
    index,reuse=s.load_index(context,dataset)
    assert isinstance(index,BM25Retriever) and reuse["tree_or_graph_built"] is False
    task="When did Aster Archive open?"
    item={"question":task,**{f"choice_{c}":c for c in "ABCD"}}
    evidence,records=retrieve_context(task,index,2)
    row=s.generate("fixture",task,index,dataset,longbench_item=item if dataset=="longbench" else None)
    messages,maximum,kwargs=client.payloads[0]
    assert row["status"]=="ok" and row["trace"]==[] and len(client.calls)==1
    assert row["evidence_records"]==records and row["termination"]=="single_final_generation"
    expected=solver_messages(task,evidence) if dataset=="math" else [{"role":"user","content":official_prompt(item,evidence,rag=True)}]
    assert messages==expected and maximum==(4096 if dataset=="math" else 128)
    assert kwargs["tag"]==("math_bm25_answer" if dataset=="math" else "bm25_answer")


def test_direct_longbench_preserves_original_full_prompt_truncation(tmp_path):
    client=Client();s=suite(tmp_path,"direct",client)
    item={"question":"When did Aster Archive open?","context":"z"*3000,**{f"choice_{c}":c for c in "ABCD"}}
    index,reuse=s.load_index(item["context"],"longbench")
    assert reuse["tree_or_graph_built"] is False and not hasattr(index,"entity_graph")
    row=s.generate("fixture","task",index,"longbench",longbench_item=item)
    prompt=official_prompt(item,item["context"]);budget=s.config["longbench_direct_input_budget"]
    messages,maximum,kwargs=client.payloads[0]
    assert messages==[{"role":"user","content":prompt[:budget//2]+prompt[-budget//2:]}]
    assert row["truncated"] and row["original_prompt_tokens"]==len(prompt)
    assert maximum==128 and kwargs["tag"]=="direct"
