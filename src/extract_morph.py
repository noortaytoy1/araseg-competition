"""Extract per-word morphological features (POS + proclitics) for morphology-
feature fusion. Targets the shared BERT blind spot: the «و»-clause-vs-list
problem is a morphology problem (is «و» a clause-level conjunction on a
verb/maSdar, or a phrase conjunction on a noun?). camel_tools' BERT
disambiguator gives POS + proclitic analysis per word. Saved per doc/split;
consumed by the fusion head in train_encoder (flag-gated).
"""
import json
import os
import pickle
import sys

from camel_tools.disambig.bert import BERTUnfactoredDisambiguator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")
CHUNK = 150

D = BERTUnfactoredDisambiguator.pretrained("msa")


def feats(tokens):
    out = []
    for s in range(0, len(tokens), CHUNK):
        chunk = tokens[s:s + CHUNK]
        for a in D.disambiguate(chunk):
            t = a.analyses[0].analysis if a.analyses else {}
            out.append((t.get("pos", "?"), t.get("prc0", "na"),
                        t.get("prc1", "na"), t.get("prc2", "na")))
    assert len(out) == len(tokens)
    return out


def main():
    tasks = sys.argv[1:] or ["NoPnx-NP"]
    for task in tasks:
        for split in ("train", "dev"):
            path = os.path.join(DATA, f"{task}_{split}.jsonl")
            docs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            res = {}
            for i, d in enumerate(docs):
                res[d["doc_id"]] = feats(d["tokens"])
                if (i + 1) % 25 == 0:
                    print(f"  {task} {split}: {i+1}/{len(docs)}")
            out = os.path.join(DATA, f"morph_{task}_{split}.pkl")
            pickle.dump(res, open(out, "wb"))
            # vocab summary
            pos = set(f[0] for v in res.values() for f in v)
            print(f"{task} {split}: {len(docs)} docs -> {out}  ({len(pos)} POS tags)")


if __name__ == "__main__":
    main()
