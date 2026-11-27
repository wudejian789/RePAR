"""Task-independent source evidence memory with stable provenance and a word budget."""


class EvidenceMemory:
    def __init__(self,budget):
        self.budget=budget
        self.records={}
        self.first_seen={}

    def observe(self,observation,step):
        added=[]
        for rank,record in enumerate(observation.get("results",[])):
            cid=record["id"]
            if cid not in self.records:
                self.first_seen[cid]=(step,rank)
                added.append(cid)
            self.records[cid]=record
        return added

    def render(self):
        ordered=sorted(self.records,key=lambda cid:(self.first_seen[cid][1],self.first_seen[cid][0],int(cid[1:])))
        selected=[];words=0
        for cid in ordered:
            record=self.records[cid];size=len(record["text"].split())
            if words+size<=self.budget:
                selected.append(record);words+=size
        selected.sort(key=lambda r:r["start"])
        return "\n\n".join(f"[{r['id']} | entities={','.join(r.get('entity_ids',[])[:12])} | chars={r['start']}:{r['end']}]\n{r['text']}"
                            for r in selected),[r["id"] for r in selected]
