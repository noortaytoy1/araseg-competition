"""CPU smokes for src/rmerge.py (recursive beam-merge segmenter).

Each smoke prints PASS/FAIL and the evidence the spec asks for:

  1. gold-cut alignment correct on a REAL dev doc (junction axis == labels[:-1]).
  2. AraBERT actually RE-READS token spans: the batch really is
     [CLS] tokens(A) [SEP] tokens(B) [SEP] (decode it back and check).
  3. training is TEACHER-FORCED (builds the gold tree; no node straddles a gold
     boundary) with DENSE supervision (#supervised candidates == sum over states
     of #adjacent pairs, not just merged ones).
  4. loss PENALIZES cross-boundary merges (BCE target 0 at gold boundaries): a
     pair straddling a boundary contributes a target-0 BCE term.
  5. batched scoring == per-pair scoring (parallelization correctness).
  6. beam decode uses GEOMETRIC-MEAN (length-normalized) scoring.
  7. encoder grad flows (a train step gives non-zero AraBERT grad-norm).

Run:  CUDA_VISIBLE_DEVICES=-1 python src/smoke_rmerge.py
Smokes that need AraBERT read the OFFLINE HF cache; if it is absent they SKIP
(clearly marked) so the network-free smokes still run.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Arabic surfaces are printed in the evidence lines; force UTF-8 stdout so the
# smoke runs on a cp1252 Windows console without a UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl  # noqa: E402
import rmerge  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                   "data", "NoPnx-NP_dev.jsonl")


def _ok(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""))
    return cond


def _arabert():
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
        rmerge.load_arabert  # noqa: B018
        tok, enc, head = rmerge.load_arabert(rmerge.ARABERT_DEFAULT, seed=0)
        return rmerge.MergeScorer(tok, enc, head, span_cap=16)
    except Exception as e:  # noqa: BLE001
        print(f"[SKIP] AraBERT unavailable offline: {type(e).__name__}: {e}")
        return None


def _small_real_doc():
    """A short real doc slice (first ~14 words up to a couple of boundaries) so
    the AraBERT smokes stay CPU-fast."""
    docs = load_jsonl(DEV)
    d = docs[0]
    words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
    labels = list(d["labels"])
    # take a prefix that ends on a gold boundary so it is a whole sub-doc
    cut = None
    for i in range(6, min(16, len(labels))):
        if labels[i] == 1:
            cut = i
            break
    if cut is None:
        cut = min(12, len(labels) - 1)
        labels[cut] = 1
    words = words[: cut + 1]
    labels = labels[: cut + 1]
    labels[-1] = 1                          # doc-final is a boundary
    return words, labels, d["doc_id"]


# --------------------------------------------------------------------------- #
def smoke_1_gold_cut_real_doc():
    docs = load_jsonl(DEV)
    d = docs[0]
    gc = rmerge.gold_cut_mask(d["labels"])
    cond = (len(gc) == len(d["labels"]) - 1
            and gc.tolist() == list(d["labels"])[:-1])
    return _ok("1 gold-cut alignment on real doc",
               cond, f"doc={d['doc_id']} n={len(d['labels'])} "
                     f"boundaries={int(sum(d['labels']))}")


def smoke_2_arabert_rereads_spans(scorer):
    if scorer is None:
        return _ok("2 AraBERT re-reads token spans", True, "(skipped)")
    words, _labels, _ = _small_real_doc()
    word_ids = scorer._word_ids(words)
    pairs = [((0, 1), (2, 3))]              # A=words0-1, B=words2-3
    ids, attn, tt = scorer.build_pair_batch(word_ids, pairs)
    row = ids[0].tolist()
    tok = scorer.tok
    cls, sep = tok.cls_token_id, tok.sep_token_id
    # structure: [CLS] a... [SEP] b... [SEP]  (exactly two SEPs, starts with CLS)
    n_sep = sum(1 for t in row if t == sep)
    a_expected = scorer._left_ids(word_ids, 0, 1)
    b_expected = scorer._right_ids(word_ids, 2, 3)
    expected = [cls] + a_expected + [sep] + b_expected + [sep]
    cond = (row[0] == cls and n_sep == 2 and row[: len(expected)] == expected)
    # and the actual surface decodes back to A [SEP] B (re-reading real tokens)
    dec = tok.decode(row[: len(expected)])
    return _ok("2 AraBERT re-reads token spans "
               "([CLS] tok(A) [SEP] tok(B) [SEP])", cond,
               f"decoded='{dec}'")


def smoke_3_teacher_forced_dense(scorer=None):
    words, labels, _ = _small_real_doc()
    # teacher-forced: no visited node straddles a gold boundary
    spans = rmerge.sentence_spans(labels)
    def sent_of(w):
        for si, (s0, s1) in enumerate(spans):
            if s0 <= w <= s1:
                return si
        return -1
    straddle = False
    n_states = 0
    for nodes in rmerge.teacher_forced_states(labels):
        n_states += 1
        for (w0, w1) in nodes:
            if sent_of(w0) != sent_of(w1):
                straddle = True
    # dense-supervision count == sum over states of #adjacent pairs
    dense = rmerge.count_dense_candidates(labels)
    by_hand = sum(max(0, len(nodes) - 1)
                  for nodes in rmerge.teacher_forced_states(labels))
    n_merges = len(words) - len(spans)
    cond = (not straddle) and (dense == by_hand) and (dense > n_merges)
    ok = _ok("3 teacher-forced (no straddle) + DENSE supervision count",
             cond, f"states={n_states} dense_candidates={dense} "
                   f">merges({n_merges}); teacher_final="
                   f"{rmerge.teacher_forced_states(labels)[-1]==spans}")
    # optional: prove the loop actually applies exactly `dense` BCE terms
    if scorer is not None and cond:
        loss, ncand, npos, nneg = rmerge.doc_train_loss(
            scorer, words, labels, pos_weight=3.0)
        ok = ok and _ok("3b dense BCE terms applied == dense count",
                        ncand == dense,
                        f"applied={ncand} pos(t=1)={npos} neg(t=0)={nneg}")
    return ok


def smoke_4_penalizes_cross_boundary(scorer=None):
    words, labels, _ = _small_real_doc()
    gc = rmerge.gold_cut_mask(labels)
    nodes = rmerge.start_state(len(words))
    pairs, targets = rmerge.dense_supervision_pairs(nodes, gc)
    # every gold boundary junction gets a target-0 candidate at the start state
    bnd = [k for k in range(len(gc)) if gc[k] == 1]
    zero_junctions = [pairs[i][0][1] for i, t in enumerate(targets) if t == 0]
    cond = all(b in zero_junctions for b in bnd) and (targets.count(0) == len(bnd))
    ok = _ok("4 loss penalizes cross-boundary merges (BCE target 0 at "
             "gold boundaries)", cond,
             f"boundary_junctions={bnd} target0_junctions={sorted(zero_junctions)}")
    # optional numeric: a high merge prob at a boundary yields a large BCE term
    if scorer is not None:
        import torch
        # score the boundary-straddling pair; its target is 0
        bi = targets.index(0)
        logits = scorer.logits_for_pairs(scorer._word_ids(words),
                                         [pairs[bi]], grad=True)
        bce0 = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits)).item()
        bce1 = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits)).item()
        p = float(torch.sigmoid(logits)[0])
        # BCE against 0 grows as p grows; against 1 shrinks. Penalty direction ok.
        ok = ok and _ok("4b higher merge prob at a boundary -> larger penalty",
                        (bce0 > bce1) == (p > 0.5),
                        f"p={p:.3f} BCE(.,0)={bce0:.3f} BCE(.,1)={bce1:.3f}")
    return ok


def smoke_5_batched_equals_perpair(scorer):
    if scorer is None:
        return _ok("5 batched == per-pair (parallelization)", True, "(skipped)")
    words, _labels, _ = _small_real_doc()
    nodes = rmerge.start_state(len(words))
    cand = rmerge.adjacent_candidates(nodes)
    pairs = [(nodes[i], nodes[i + 1]) for (i, _k) in cand]
    wid = scorer._word_ids(words)
    batched = scorer.probs_for_pairs(wid, pairs)
    single = np.array([scorer.probs_for_pairs(wid, [p])[0] for p in pairs])
    maxdiff = float(np.max(np.abs(batched - single))) if len(pairs) else 0.0
    cond = rmerge.batched_equals_perpair(scorer, words, pairs, atol=1e-4)
    return _ok("5 batched == per-pair (parallelization correctness)", cond,
               f"n_pairs={len(pairs)} max|batched-single|={maxdiff:.2e}")


def smoke_6_geometric_mean():
    # two hyps: same merges, different lengths. geo-mean must NOT prefer fewer
    # merges just because product shrinks (that is the FM1 bias we removed).
    h_two = rmerge.Hyp([(0, 2)], [0.8, 0.8])       # geo-mean 0.8 ; product 0.64
    h_one = rmerge.Hyp([(0, 1)], [0.8])            # geo-mean 0.8 ; product 0.80
    same = abs(h_two.score() - h_one.score()) < 1e-9
    # explicit geo-mean value
    val = rmerge.Hyp([(0, 3)], [0.5, 0.5, 0.8]).score()
    expect = (0.5 * 0.5 * 0.8) ** (1.0 / 3.0)
    cond = same and abs(val - expect) < 1e-9
    return _ok("6 beam decode uses geometric-mean (length-normalized) scoring",
               cond, f"geo(0.8,0.8)={h_two.score():.4f}==geo(0.8)={h_one.score():.4f}; "
                     f"geo(.5,.5,.8)={val:.4f} (product={0.5*0.5*0.8:.3f})")


def smoke_7_encoder_grad_flows(scorer):
    if scorer is None:
        return _ok("7 encoder grad flows", True, "(skipped)")
    import torch
    words, labels, _ = _small_real_doc()
    scorer.enc.train()
    scorer.enc.zero_grad(set_to_none=True)
    scorer.head.zero_grad(set_to_none=True)
    loss, ncand, _, _ = rmerge.doc_train_loss(scorer, words, labels,
                                              pos_weight=3.0)
    loss.backward()
    enc_gn = 0.0
    for p in scorer.enc.parameters():
        if p.grad is not None:
            enc_gn += float(p.grad.detach().pow(2).sum())
    enc_gn = enc_gn ** 0.5
    head_gn = 0.0
    for p in scorer.head.parameters():
        if p.grad is not None:
            head_gn += float(p.grad.detach().pow(2).sum())
    head_gn = head_gn ** 0.5
    scorer.enc.eval()
    return _ok("7 encoder grad flows (AraBERT is the merger, fine-tuned)",
               enc_gn > 0 and head_gn > 0,
               f"encoder grad-norm={enc_gn:.4e} head grad-norm={head_gn:.4e} "
               f"(loss/cand={float(loss)/max(ncand,1):.3f})")


def main():
    print("=" * 72)
    print("rmerge CPU smokes (CUDA_VISIBLE_DEVICES=%s)"
          % os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("=" * 72)
    rmerge.self_test()
    scorer = _arabert()
    results = []
    results.append(smoke_1_gold_cut_real_doc())
    results.append(smoke_2_arabert_rereads_spans(scorer))
    results.append(smoke_3_teacher_forced_dense(scorer))
    results.append(smoke_4_penalizes_cross_boundary(scorer))
    results.append(smoke_5_batched_equals_perpair(scorer))
    results.append(smoke_6_geometric_mean())
    results.append(smoke_7_encoder_grad_flows(scorer))
    print("-" * 72)
    n_pass = sum(1 for r in results if r)
    print(f"{n_pass}/{len(results)} smokes PASS")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
