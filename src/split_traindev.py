"""Fold dev into training (organizer-sanctioned). Hold out 20% of dev to
measure whether the extra data helps; train+80%dev is the treatment set."""
import os, sys
task = sys.argv[1]
D = os.path.join(os.path.dirname(__file__), os.pardir, "data")
tr = [l for l in open(os.path.join(D, f"{task}_train.jsonl"), encoding="utf-8") if l.strip()]
dv = [l for l in open(os.path.join(D, f"{task}_dev.jsonl"), encoding="utf-8") if l.strip()]
nheld = max(1, len(dv) // 5)
held, dvtrain = dv[-nheld:], dv[:-nheld]
open(os.path.join(D, f"{task}_trdev_train.jsonl"), "w", encoding="utf-8").writelines(tr + dvtrain)
open(os.path.join(D, f"{task}_devheld.jsonl"), "w", encoding="utf-8").writelines(held)
print(f"{task}: train {len(tr)} + dev-train {len(dvtrain)} = {len(tr)+len(dvtrain)} docs | held-out {len(held)}")
