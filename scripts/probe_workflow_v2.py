"""Real-model protocol checks on synthetic general tasks, without benchmark inputs."""
from repar.client import LLMClient
from repar.index_v2 import KnowledgeIndex
from repar.io import read_json,write_json
from repar.workflow import research

c=read_json("configs/repar_v2.json")
c["chunk_words"]=45
context=("The Delta Commission developed the Silver Protocol to monitor river pollution. "
         "The first pilot used laboratory sensors and independent technical reviewers.\n\n"
         "The museum holds paintings and architectural drawings. Visitors can enter on weekdays. "
         "The annual exhibition includes regional artists and bronze sculptures.\n\n"
         "Harbor Council adopted the Silver Protocol in 2024. Harbor Council funds mobile clinics. "
         "Mobile clinics reduce travel costs for residents.\n\n"
         "North Agency did not adopt the Silver Protocol in 2023. North Agency may fund pilot programs.")
index=KnowledgeIndex(context,c)
tasks=["Which organization adopted the protocol developed by the Delta Commission, and when?",
       "Summarize what the source says about museum visiting times.",
       "Did North Agency adopt the protocol in 2023? Ground your response in the source."]
rows=[]
for task in tasks:
    client=LLMClient(c,cache_dir="artifacts/cache/workflow_v2_probe")
    result=research(task,index,client,c)
    row={"task":task,"termination":result.termination,"evidence_ids":result.evidence_ids,
         "trace":result.trace,"calls":len(client.calls)}
    rows.append(row)
    print(task,result.termination,[(s.get('plan'),[x.get('name') for x in s.get('executed',[])]) for s in result.trace],flush=True)
write_json("artifacts/workflow_v2_probe.json",{"config":c,"rows":rows})
