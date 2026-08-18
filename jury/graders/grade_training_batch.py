import json,sys,argparse
# grade4.py — retrain4 variant of grade3.py: identical grading, paths under scratch_exo/retrain4/. Gold = train jsonl only.
import numpy as np
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ap=argparse.ArgumentParser();ap.add_argument("--track",required=True);ap.add_argument("--juror",required=True);ap.add_argument("--only",required=True)
a=ap.parse_args()
GOLD={"nonp":"data/NoPnx-NP_train.jsonl","nopa":"data/NoPnx-PA_train.jsonl","pnxnp":"data/NP_train.jsonl"}[a.track]
HASNL=(a.track=="nopa")
tr={}
for l in open(GOLD,encoding="utf-8"):
    if l.strip():
        d=json.loads(l);tr[d["doc_id"]]=d
ids=[x for x in a.only.split(",") if x]
def snap(T,i,w):
    n=len(T)
    if w and 0<=i<n and T[i]==w: return i
    if w:
        near=[k for k in range(max(0,i-2),min(n,i+3)) if T[k]==w]
        if near: return min(near,key=lambda x:abs(x-i))
    return i if 0<=i<n else None
def ctx(T,row,i):
    lo,hi=max(0,i-9),min(len(T),i+10);parts=[]
    for k in range(lo,hi):
        w="<NL>" if T[k]=="\n" else T[k]
        if k==i: w="[["+w+"]]"
        parts.append(w)
        if row[k]==1: parts.append("‖")
    return " ".join(parts)
rep=[f"=== MISTAKES — track {a.track} jury {a.juror} ==="]
means=[]
for did in ids:
    d=tr[did];T=d["tokens"];g=np.array(d["labels"])
    v=json.load(open(f"scratch_exo/retrain4/{a.track}/ans/j{a.juror}/{did}.json",encoding="utf-8"))
    row=np.zeros(len(T),int)
    for b in v.get("boundaries",[]):
        k=snap(T,int(b["i"]),b.get("w",""))
        if k is not None and T[k]!="\n": row[k]=1
    row[-1]=1
    if HASNL:
        for i,t in enumerate(T):
            if t=="\n":
                row[i]=0
                if i>0 and T[i-1]!="\n": row[i-1]=1
    tp=int(((row==1)&(g==1)).sum());fp=int(((row==1)&(g==0)).sum());fn=int(((row==0)&(g==1)).sum())
    P=100*tp/max(tp+fp,1);R=100*tp/max(tp+fn,1);F=100*2*tp/max(2*tp+fp+fn,1)
    means.append(F)
    print(f"{did}: F1 {F:.1f} (P {P:.1f} / R {R:.1f})")
    rep.append(f"--- {did}: F1 {F:.1f} ---")
    for i in range(len(T)):
        if row[i]==1 and g[i]==0:
            rep.append(f"  FALSE CUT @{i}:")
            rep.append(f"    yours: ...{ctx(T,row,i)}...")
            rep.append(f"    gold:  ...{ctx(T,g,i)}...")
        elif row[i]==0 and g[i]==1:
            rep.append(f"  MISSED @{i}:")
            rep.append(f"    yours: ...{ctx(T,row,i)}...")
            rep.append(f"    gold:  ...{ctx(T,g,i)}...")
open(f"scratch_exo/retrain4/{a.track}/mistakes_j{a.juror}.txt","w",encoding="utf-8").write(chr(10).join(rep))
print(f"BATCH MEAN F1: {sum(means)/max(len(means),1):.2f}")
print("GRADING_DONE")
