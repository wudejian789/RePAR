"""Build every entity graph without questions, choices or labels entering the index."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import time

from repar.index_v2 import KnowledgeIndex
from repar.io import load_public_samples,read_json,write_json


def build_one(args):
    number,item,config=args
    started=time.perf_counter()
    print(f"START {number}/125 {item['_id']} chars={len(item['context'])}",flush=True)
    index=KnowledgeIndex.cached(item["context"],config)
    print(f"DONE {number}/125 seconds={time.perf_counter()-started:.2f} entities={index.stats['entities']} relations={index.stats['relations']}",flush=True)
    return {"_id":item["_id"],**index.stats}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    config=read_json("configs/repar_v2.json")
    samples=load_public_samples("data/longbench_v2.json")
    rows=[]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(build_one,((n,item,config) for n,item in enumerate(samples,1))):
            rows.append(row)
            write_json("artifacts/indexes_v2/build_progress.json",{"completed":len(rows),"total":125,"indexes":rows})
    write_json("artifacts/indexes_v2/build_complete.json",{"completed":125,"indexes":rows,
                                                         "resume_wall_seconds":time.perf_counter()-started,
                                                         "workers":args.workers})


if __name__=="__main__":main()
