"""Read-only coverage for the paired backbone run; no partial accuracy claims."""
from datetime import datetime
from pathlib import Path
import argparse
import json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/backbone_27b_comparison.json")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/args.config).read_text(encoding="utf-8"))
    conditions=[]
    for filename in config["conditions"]:
        c=json.loads((root/filename).read_text(encoding="utf-8"));p=root/c["output"]
        item={"condition":c["method_display"]}
        for dataset,name,expected in (("math","predictions",19),("math","grades",19),("longbench","predictions",125)):
            f=p/dataset/(name+".jsonl")
            records=[json.loads(s) for s in f.read_text(encoding="utf-8").split("\n") if s.strip()] if f.exists() else []
            item[dataset+"_"+name]={"rows":len(records),"unique":len({r["id"] for r in records}),
                "expected":expected,"errors":sum(r.get("status")=="error" for r in records)}
        item["summary_complete"]=(p/"summary.json").exists()
        item["audit_passed"]=json.loads((p/"audit.json").read_text(encoding="utf-8")).get("passed",False) if (p/"audit.json").exists() else False
        conditions.append(item)
    print(json.dumps({"time":datetime.now().astimezone().isoformat(),"conditions":conditions},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
