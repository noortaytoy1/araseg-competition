"""Closed-track data multiplication via COHERENT contiguous-block recombination.

This is the structure-preserving FIX for make_recomb.py's -1.56 failure.

make_recomb.py concatenated RANDOM single sentences drawn from the whole corpus
up to a FIXED median length. That produced register-salad: globally incoherent
documents flattened to one length, which scored -1.56 by going out-of-distribution
on BOTH axes (topical coherence AND the per-doc length profile).

This module instead assembles every synthetic document from a CONTIGUOUS run of
consecutive sentences taken from a SINGLE source train document, so local topical
and register coherence is preserved by construction, and it draws each synthetic
document's length from the REAL per-doc length histogram (heavy right tail
included) rather than collapsing everything to the median.

CLOSED-TRACK LAW (non-negotiable, enforced here):
  * The ONLY file read is data/{TASK}_train.jsonl (the 174 train docs).
  * Nothing is ever read, sampled, or fit from dev/test or any external corpus.
  * Every token in every synthetic doc is copied verbatim from a train doc.
  * Boundary labels are correct BY CONSTRUCTION: a synthetic doc is built only
    out of whole gold sentences, so each internal boundary is a real gold
    boundary and the final token is always a real sentence end (label == 1).
    Nothing is ever self-segmented or guessed.

Output schema mirrors make_recomb.py EXACTLY:
  * originals are passed through verbatim (all their original keys survive);
  * each synthetic doc carries only {"doc_id", "tokens", "labels"}.

Usage:
    python block_aug.py <task> <multiplier>
        <task>        one of the AraSeg variants (default NoPnx-NP)
        <multiplier>  N synthetic docs per original (default 5 -> 174*5 = 870)

Writes:  data/block_{TASK}_train.jsonl  =  originals + N x synthetic docs.

Self-test (CPU / tiny scale only):
    python block_aug.py --selftest
    python block_aug.py --smoke <task>
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Reproducibility: fixed seed, same discipline as make_recomb.py (seed 42).
# ---------------------------------------------------------------------------
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")

Sentence = Tuple[List[str], List[int]]  # (tokens, labels) of ONE gold sentence


# ---------------------------------------------------------------------------
# 1. Split each source doc into its gold sentences, remembering provenance.
#    A "sentence" is the contiguous span from just-after the previous boundary
#    up to and including the next label==1 token. A trailing span with no final
#    boundary (never present in the real train files, but handled defensively)
#    is kept as-is and NOT relied on to end a synthetic doc.
# ---------------------------------------------------------------------------
def split_sentences(doc: dict) -> List[Sentence]:
    """Return the ordered list of gold sentences of one document.

    Boundaries are read straight from the gold labels; nothing is inferred.
    """
    toks, labs = doc["tokens"], doc["labels"]
    out: List[Sentence] = []
    start = 0
    for i, l in enumerate(labs):
        if l == 1:
            out.append((list(toks[start:i + 1]), list(labs[start:i + 1])))
            start = i + 1
    if start < len(toks):  # trailing non-boundary span (defensive; ~never happens)
        out.append((list(toks[start:]), list(labs[start:])))
    return out


def build_doc_sentences(docs: Sequence[dict]) -> List[List[Sentence]]:
    """One list of gold sentences per source doc (provenance preserved)."""
    return [split_sentences(d) for d in docs]


def real_length_histogram(docs: Sequence[dict]) -> List[int]:
    """The empirical per-doc token-length list of the real train docs.

    Sampling a target length uniformly from THIS list reproduces the real
    length distribution -- median, both shoulders, AND the heavy right tail
    (the 15k-token documents) -- by construction, instead of collapsing to a
    single median as make_recomb.py did.
    """
    return [len(d["tokens"]) for d in docs]


# ---------------------------------------------------------------------------
# 2. Assemble ONE synthetic doc: a contiguous run of consecutive sentences
#    from a SINGLE source doc, sized to a target length drawn from the real
#    per-doc length histogram, optionally stitching 1-2 further contiguous
#    blocks FROM THE SAME source doc when one run cannot reach a long target.
# ---------------------------------------------------------------------------
def _sentence_ends_on_boundary(sent: Sentence) -> bool:
    return len(sent[1]) > 0 and sent[1][-1] == 1


def _doc_len(sents: List[Sentence]) -> int:
    return sum(len(st) for st, _ in sents)


def _contiguous_run(sents: List[Sentence], target: int, start: int,
                    ) -> Tuple[List[str], List[int], int]:
    """Take a contiguous run of consecutive sentences starting at index `start`,
    growing to the boundary at or just past `target` tokens.

    Extend forward one whole sentence at a time. Stop at the FIRST sentence
    whose inclusion reaches or exceeds `target` (or the last available
    sentence). Because every appended unit is a whole gold sentence ending on
    label==1, the run's last token is always a real gold boundary. Returns
    (tokens, labels, index_after_last_used_sentence).
    """
    n = len(sents)
    toks: List[str] = []
    labs: List[int] = []
    i = start
    while i < n:
        st, sl = sents[i]
        toks += st
        labs += sl
        i += 1
        if len(toks) >= target:
            break
    # Boundary guarantee: real sentences all end on label==1, so labs[-1]==1
    # already. Defensive walk-back only for a (never-seen) trailing span.
    if labs and labs[-1] != 1:
        last_b = max((j for j, l in enumerate(labs) if l == 1), default=-1)
        toks, labs = (toks[:last_b + 1], labs[:last_b + 1]) if last_b >= 0 else ([], [])
    return toks, labs, i


def _start_index_for_target(sents: List[Sentence], target: int,
                            rng: random.Random) -> int:
    """Choose a start sentence so a forward run can plausibly reach `target`.

    Restrict the start to sentences that leave >= target tokens of forward
    material ahead when the doc is long enough; this removes the "half the
    doc" truncation that a fully-random start causes. When the doc is only
    just long enough, this collapses to starting at (or near) sentence 0.
    """
    n = len(sents)
    # cumulative tokens from each start index to the end of the doc
    suffix = 0
    forward = [0] * n
    for j in range(n - 1, -1, -1):
        suffix += len(sents[j][0])
        forward[j] = suffix
    valid = [j for j in range(n) if forward[j] >= target]
    if not valid:            # target exceeds whole doc: start at 0
        return 0
    return rng.choice(valid)


def make_synthetic_doc(sents: List[Sentence], target: int,
                       rng: random.Random, doc_id: str,
                       max_blocks: int = 2) -> dict:
    """Build one coherent-block synthetic doc of ~`target` tokens from ONE
    already-chosen source doc's sentence list `sents`.

    A contiguous run of consecutive sentences is taken from a start point that
    leaves enough forward material. Because the source doc AND the target
    length are bound together by the caller (target is drawn near this doc's
    own length), the block inherits this doc's register -- crucially its
    sentence-length regime, so the boundary rate matches by construction rather
    than being blended across registers. If the run still falls short (target
    jittered above this doc's length), up to `max_blocks-1` further CONTIGUOUS
    blocks FROM THE SAME doc are stitched on; still one document, coherence and
    register preserved.
    """
    start = _start_index_for_target(sents, target, rng)
    toks, labs, _ = _contiguous_run(sents, target, start)

    blocks = 1
    while len(toks) < 0.9 * target and blocks < max_blocks:
        remaining = target - len(toks)
        s2 = _start_index_for_target(sents, remaining, rng)
        bt, bl, _ = _contiguous_run(sents, remaining, s2)
        if not bt:
            break
        toks += bt
        labs += bl
        blocks += 1

    # Final safety: a synthetic doc MUST end on a boundary. By construction it
    # does; guard defensively against a source with only a trailing span.
    if not labs or labs[-1] != 1:
        for s in sents:
            if _sentence_ends_on_boundary(s):
                st, sl = s
                toks, labs = list(st), list(sl)
                break

    return {"doc_id": doc_id, "tokens": toks, "labels": labs}


# ---------------------------------------------------------------------------
# 3. Driver: originals (verbatim) + N x synthetic docs -> block_{TASK}_train.jsonl
# ---------------------------------------------------------------------------
def generate(docs: Sequence[dict], mult: int,
             rng: random.Random) -> List[dict]:
    """Return originals (passed through verbatim) followed by mult*len(docs)
    coherent-block synthetic docs.

    Source-and-length are BOUND: each synthetic doc picks one source train doc
    and draws its target length near THAT doc's own length. This reproduces the
    real per-doc length histogram (heavy tail included -- long docs are sampled
    as sources and yield long synthetic docs) AND preserves each register's
    sentence-length regime, so the corpus boundary rate reproduces itself
    instead of being blended OOD across registers.
    """
    doc_sents = build_doc_sentences(docs)
    lengths = real_length_histogram(docs)
    lo, hi = min(lengths), max(lengths)

    out: List[dict] = list(docs)  # originals kept verbatim (all keys survive)
    n_synth = mult * len(docs)
    for k in range(n_synth):
        # Pick a source doc (uniform over the 174, so the length histogram of
        # the *sources* equals the real length histogram). Draw the target
        # near that doc's own length with mild jitter, so we do not merely
        # replicate the 174 exact lengths but stay in the doc's own regime.
        src = rng.randrange(len(doc_sents))
        sents = doc_sents[src]
        base = _doc_len(sents)
        jitter = rng.uniform(0.85, 1.15)
        target = int(round(base * jitter))
        target = max(lo, min(target, hi))
        out.append(make_synthetic_doc(sents, target, rng,
                                      doc_id=f"block_{k}"))
    return out


def load_train(task: str) -> List[dict]:
    """Load ONLY the closed-track train split. No other split is ever touched."""
    path = os.path.join(DATA, f"{task}_train.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_docs(docs: Sequence[dict], dst: str) -> None:
    with open(dst, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else "NoPnx-NP"
    mult = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rng = random.Random(SEED)

    docs = load_train(task)
    out = generate(docs, mult, rng)

    dst = os.path.join(DATA, f"block_{task}_train.jsonl")
    write_docs(out, dst)

    synth = [d for d in out if str(d["doc_id"]).startswith("block_")]
    bad = sum(1 for d in synth if not d["labels"] or d["labels"][-1] != 1)
    pool = sum(len(split_sentences(d)) for d in docs)
    print(f"{len(docs)} orig + {mult}x = {len(out)} docs | "
          f"{pool} sentence pool | "
          f"synth-docs not ending on boundary: {bad} -> {dst}")


# ===========================================================================
# Self-tests (CPU / tiny scale ONLY -- no GPU, no full-scale generation).
# ===========================================================================
def _profile(docs: Sequence[dict]) -> Dict[str, float]:
    L = sorted(len(d["tokens"]) for d in docs)
    pos = sum(sum(d["labels"]) for d in docs)
    tot = sum(len(d["labels"]) for d in docs)
    q = lambda p: L[int(p * (len(L) - 1))]
    return {
        "n": len(docs),
        "min": L[0], "p25": q(.25), "median": statistics.median(L),
        "p75": q(.75), "p90": q(.90), "p99": q(.99), "max": L[-1],
        "mean": statistics.mean(L),
        "boundary_rate": pos / max(tot, 1),
    }


def _fmt(p: Dict[str, float]) -> str:
    return (f"n={p['n']:>4}  len min/p25/med/p75/p90/p99/max = "
            f"{p['min']}/{p['p25']}/{p['median']:.0f}/{p['p75']}/"
            f"{p['p90']}/{p['p99']}/{p['max']}  "
            f"mean={p['mean']:.0f}  brate={p['boundary_rate']:.4f}")


def _validate_labels(docs: Sequence[dict], sentence_set) -> Tuple[int, int, int]:
    """Return (n_bad_end, n_bad_internal, n_untraceable) over synthetic docs.

    * bad_end       : synthetic doc whose last label != 1.
    * bad_internal  : synthetic doc whose (tokens,labels) does not decompose
                      into a concatenation of WHOLE gold sentences from train
                      (i.e. an invented boundary or a broken one).
    * untraceable   : a token in a synthetic doc absent from any train doc's
                      sentence pool (closed-track violation).
    """
    bad_end = bad_internal = untraceable = 0
    for d in docs:
        labs = d["labels"]
        if not labs or labs[-1] != 1:
            bad_end += 1
        # Re-derive the sentences of this synthetic doc and confirm each is a
        # verbatim gold sentence from the pool.
        derived = split_sentences(d)
        for st, sl in derived:
            key = (tuple(st), tuple(sl))
            if key not in sentence_set:
                bad_internal += 1
                break
    return bad_end, bad_internal, untraceable


def selftest() -> None:
    """Tiny synthetic-corpus test: no real data, no GPU. Proves label validity,
    boundary-by-construction, and round-trip integrity on a toy corpus."""
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(30)]

    def toy_doc(doc_id: str, n_sent: int) -> dict:
        toks: List[str] = []
        labs: List[int] = []
        for _ in range(n_sent):
            slen = rng.randint(1, 6)
            for _ in range(slen):
                toks.append(rng.choice(vocab))
                labs.append(0)
            labs[-1] = 1  # sentence boundary
        return {"doc_id": doc_id, "tokens": toks, "labels": labs,
                "text": " ".join(toks)}

    docs = [toy_doc(f"toy_{i}", rng.randint(3, 20)) for i in range(12)]

    # sentence pool for traceability check
    pool = set()
    for d in docs:
        for st, sl in split_sentences(d):
            pool.add((tuple(st), tuple(sl)))

    out = generate(docs, mult=5, rng=random.Random(1))
    synth = [d for d in out if str(d["doc_id"]).startswith("block_")]

    # (a) label validity
    bad_end, bad_internal, _ = _validate_labels(synth, pool)
    assert bad_end == 0, f"{bad_end} synth docs not ending on a boundary"
    assert bad_internal == 0, f"{bad_internal} synth docs with invented/broken boundaries"

    # (b) every synth doc is non-empty and internally consistent
    for d in synth:
        assert len(d["tokens"]) == len(d["labels"]) > 0
        assert set(d["labels"]) <= {0, 1}

    # (c) round-trip: write -> read -> identical
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "rt.jsonl")
        write_docs(out, p)
        back = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    assert len(back) == len(out)
    for a, b in zip(out, back):
        assert a["tokens"] == b["tokens"] and a["labels"] == b["labels"]

    # (d) originals passed through verbatim, keys intact
    orig_back = [d for d in out if not str(d["doc_id"]).startswith("block_")]
    assert len(orig_back) == len(docs)
    assert "text" in orig_back[0], "original doc lost its extra keys"
    # synthetic docs carry EXACTLY make_recomb's 3 keys
    assert set(synth[0].keys()) == {"doc_id", "tokens", "labels"}

    # (e) distribution match on a LARGER two-regime toy corpus: short docs with
    #     short sentences (high boundary rate) + long docs with long sentences
    #     (low boundary rate). The synthetic boundary rate and median length
    #     must track the real ones -- the register-binding property that fixed
    #     the -1.56 OOD failure -- not merely be "some" rate.
    rng2 = random.Random(7)

    def regime_doc(doc_id, n_sent, s_lo, s_hi):
        t, l = [], []
        for _ in range(n_sent):
            for _ in range(rng2.randint(s_lo, s_hi)):
                t.append(rng2.choice(vocab)); l.append(0)
            l[-1] = 1
        return {"doc_id": doc_id, "tokens": t, "labels": l}

    big = ([regime_doc(f"s{i}", rng2.randint(8, 20), 2, 5) for i in range(60)] +
           [regime_doc(f"l{i}", rng2.randint(40, 90), 10, 20) for i in range(40)])
    real_bp = _profile(big)
    synth_big = [d for d in generate(big, mult=8, rng=random.Random(9))
                 if str(d["doc_id"]).startswith("block_")]
    synth_bp = _profile(synth_big)
    assert abs(synth_bp["boundary_rate"] - real_bp["boundary_rate"]) < 0.01, \
        (f"boundary-rate OOD: synth {synth_bp['boundary_rate']:.4f} vs "
         f"real {real_bp['boundary_rate']:.4f}")
    assert 0.75 * real_bp["median"] <= synth_bp["median"] <= 1.25 * real_bp["median"], \
        f"median length OOD: synth {synth_bp['median']} vs real {real_bp['median']}"
    # tail must survive: synth max reaches the real regime, not collapsed
    assert synth_bp["max"] >= 0.6 * real_bp["max"], "heavy tail collapsed"

    print("SELFTEST OK: label validity, boundary-by-construction, round-trip, "
          "schema mirror, and distribution match (length + boundary rate + tail) "
          "all pass on the toy corpora.")


def smoke(task: str, mult: int = 5) -> None:
    """Real-train smoke report (READS ONLY train). Prints the generated
    length/boundary profile against the real train profile plus examples."""
    # Windows consoles default to cp1252 and choke on Arabic; force UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # py3.7+
    except Exception:
        pass
    rng = random.Random(SEED)
    docs = load_train(task)

    # sentence pool for traceability / boundary-validity check
    pool = set()
    for d in docs:
        for st, sl in split_sentences(d):
            pool.add((tuple(st), tuple(sl)))

    out = generate(docs, mult, rng)
    synth = [d for d in out if str(d["doc_id"]).startswith("block_")]

    real_p = _profile(docs)
    synth_p = _profile(synth)
    all_p = _profile(out)

    bad_end, bad_internal, _ = _validate_labels(synth, pool)

    print("=" * 78)
    print(f"SMOKE REPORT  --  block_aug  task={task}  mult={mult}  (READ ONLY train)")
    print("=" * 78)
    print(f"  source train docs      : {len(docs)}")
    print(f"  sentence pool          : {len(pool)} distinct gold sentences")
    print(f"  synthetic docs         : {len(synth)}")
    print(f"  output docs (orig+syn) : {len(out)}")
    print("-" * 78)
    print(f"  REAL  train  : {_fmt(real_p)}")
    print(f"  SYNTH block  : {_fmt(synth_p)}")
    print(f"  ALL (o+syn)  : {_fmt(all_p)}")
    print("-" * 78)
    print(f"  boundary-rate  real={real_p['boundary_rate']:.4f}  "
          f"synth={synth_p['boundary_rate']:.4f}  "
          f"delta={synth_p['boundary_rate'] - real_p['boundary_rate']:+.4f}")
    tail_real = sum(1 for d in docs if len(d["tokens"]) > real_p["p90"])
    tail_syn = sum(1 for d in synth if len(d["tokens"]) > real_p["p90"])
    print(f"  tail (> real p90={real_p['p90']}): "
          f"real {tail_real}/{len(docs)}={tail_real/len(docs):.1%}  "
          f"synth {tail_syn}/{len(synth)}={tail_syn/len(synth):.1%}")
    print("-" * 78)
    print(f"  VALIDITY  synth not ending on boundary : {bad_end}")
    print(f"            synth with invented/broken boundary : {bad_internal}")
    print("-" * 78)
    # two example synthetic docs (a short one and a long one)
    ex_sorted = sorted(synth, key=lambda d: len(d["tokens"]))
    for tag, d in [("SHORT", ex_sorted[len(ex_sorted) // 10]),
                   ("LONG ", ex_sorted[-1])]:
        nsent = sum(d["labels"])
        head = " ".join(d["tokens"][:16])
        print(f"  [{tag}] id={d['doc_id']}  len={len(d['tokens'])}  "
              f"sentences={nsent}  ends_on_boundary={d['labels'][-1] == 1}")
        print(f"         first16: {head} ...")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        t = sys.argv[2] if len(sys.argv) > 2 else "NoPnx-NP"
        m = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        smoke(t, m)
    else:
        main()
