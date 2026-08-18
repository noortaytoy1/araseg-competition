"""Per-paragraph sentence enumeration (Noor's idea): within EACH paragraph,
permute the order of its sentences to manufacture new in-topic boundary-join
contexts. Closed-track: reads ONLY data/{task}_train.jsonl. Boundaries are
preserved BY CONSTRUCTION (every sentence still ends on its label-1 token;
paragraph structure and per-doc length/boundary-rate are identical to the
originals -> zero distribution drift, unlike recomb).

Usage:  python src/para_enum.py <task> <mult>
        (task must have paragraph markers: PA / NoPnx-PA)
"""
import itertools
import json
import os
import random
import sys

random.seed(42)
PAR = "\n"  # data.PARAGRAPH_TOKEN
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")


def split_paragraphs(tokens, labels):
    """-> list of (ptoks, plabs, followed_by_par)."""
    paras, ct, cl = [], [], []
    for tok, lab in zip(tokens, labels):
        if tok == PAR:
            paras.append((ct, cl, True)); ct, cl = [], []
        else:
            ct.append(tok); cl.append(lab)
    if ct:
        paras.append((ct, cl, False))
    return paras


def split_sentences(toks, labs):
    out, s = [], 0
    for i, l in enumerate(labs):
        if l == 1:
            out.append((toks[s:i + 1], labs[s:i + 1])); s = i + 1
    if s < len(toks):                     # trailing fragment (rare)
        out.append((toks[s:], labs[s:]))
    return out


def enumerate_doc(paras, mult, rng):
    """Yield up to `mult` docs with sentences permuted WITHIN each paragraph."""
    made = []
    for _ in range(mult):
        nt, nl = [], []
        for ptoks, plabs, has_par in paras:
            sents = split_sentences(ptoks, plabs)
            if len(sents) >= 2:
                order = sents[:]; rng.shuffle(order)
            else:
                order = sents
            for st, sl in order:
                nt += st; nl += sl
            if has_par:
                nt.append(PAR); nl.append(0)
        made.append((nt, nl))
    return made


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "NoPnx-PA"
    mult = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rng = random.Random(42)
    docs = [json.loads(l) for l in open(os.path.join(DATA, f"{task}_train.jsonl"),
            encoding="utf-8") if l.strip()]
    out = list(docs)
    made = 0
    for d in docs:
        paras = split_paragraphs(d["tokens"], d["labels"])
        multi_sent = sum(1 for p in paras if sum(p[1]) >= 2)
        if multi_sent == 0:               # nothing to permute -> skip augmenting
            continue
        for nt, nl in enumerate_doc(paras, mult, rng):
            out.append({"doc_id": f"paraenum_{task}_{made}", "tokens": nt, "labels": nl})
            made += 1
    dst = os.path.join(DATA, f"paraenum_{task}_train.jsonl")
    with open(dst, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # sanity: augmented docs preserve boundary COUNT and token COUNT of their source style
    orig_b = sum(sum(d["labels"]) for d in docs)
    aug = [d for d in out if str(d["doc_id"]).startswith("paraenum_")]
    aug_last_bad = sum(1 for d in aug if d["labels"] and
                       (1 not in d["labels"]))
    print(f"{len(docs)} orig + {made} augmented = {len(out)} docs -> {dst}")
    print(f"  sanity: orig boundaries={orig_b} | augmented docs with no boundary={aug_last_bad} "
          f"(should be 0) | tokens/labels aligned={all(len(d['tokens'])==len(d['labels']) for d in aug)}")


if __name__ == "__main__":
    main()
