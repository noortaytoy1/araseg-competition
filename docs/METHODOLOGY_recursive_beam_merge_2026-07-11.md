# Recursive Beam-Merge Sentence Segmentation — Methodology & Failure Analysis

*AraSeg 2026, closed track (NoPnx-NP). Draft for review + researcher hand-off. 2026-07-11.*

---

## 1. Reframing (the one idea)

Sentence segmentation is **not** local boundary classification. It is **recursive sentence
construction**: a single shared merge model composes adjacent constituents bottom-up, and a
**sentence boundary is simply a junction where composition is not supported**. Gold boundaries
*supervise the merge decisions*; the segmentation is the **emergent forest** — one tree per
sentence — not an independent per-token label.

> The model is trained to learn *how language composes into sentences*. Segmentation is the
> observable shadow of that learned compositional process.

---

## 2. Architecture (formal)

### 2.1 Representation & gold alignment
- Document = words `w₁ … wₙ`. **Junction `k`** = the gap after word `k` (`k = 1 … n-1`).
- **State** = a partition of the words into contiguous **constituents (nodes)**; a **CUT** sits at
  every junction between adjacent nodes.
- **Start state** = every word is its own node → all `n-1` cuts present.
- **Gold CUT mask**: `gold_cut[k] = 1` if `k` is a true sentence boundary, else `0`. Indexed on the
  **same junction axis** as the model's cuts, so any merge the model considers at junction `k`
  aligns directly with `gold_cut[k]`. (This is the alignment step: convert the dataset's per-word
  0/1 labels into a cut mask over junctions.)

### 2.2 The merge scorer — AraBERT *is* the merger
For two adjacent nodes `A` (words `a₀..a₁`) and `B` (words `a₁+1..b₁`), the merge is scored by
**re-reading their actual tokens** through AraBERT:

```
input  =  [CLS] tokens(A) [SEP] tokens(B) [SEP]
p      =  sigmoid( w · h_[CLS] + b )   ∈ (0,1)   =  P(A and B are the same sentence)
```

- The **same** AraBERT (shared weights) scores every merge at **every level** — that is the
  recursion. Whole model fine-tuned end-to-end.
- No mean-pooling, no stored constituent vector, no curriculum, no separate composer network:
  AraBERT re-reads the real tokens each time.

### 2.3 The search — beam width 3 (training **and** inference)
- A **hypothesis** = a partial forest (node list) + a **score = product of the merge probabilities**
  that built it.
- **Each step:** for every hypothesis in the beam, score all adjacent candidate merges (**one
  batched AraBERT pass** over every candidate pair across all hypotheses); expand each hypothesis
  by applying candidate merges (erase that cut); keep the **top-3** resulting hypotheses by product
  score.
- **Stop** a hypothesis when its best available merge falls below threshold `τ` — all remaining
  junctions are boundaries. Output the **highest-product complete forest**; its surviving cuts are
  the sentence boundaries.
- Beam (vs greedy) carries 3 competing hypotheses so an early, locally-best-but-wrong merge can
  lose to a better whole-tree later → escapes greedy local minima.

### 2.4 The loss — punish cross-boundary merges
For **every** candidate merge scored at junction `k`:

```
target t_k = 1   if gold_cut[k] = 0   (same sentence → SHOULD merge)
target t_k = 0   if gold_cut[k] = 1   (boundary       → should NOT merge)

L = Σ  BCE( p_k , t_k )     over all candidates, all hypotheses, all steps
```

- **Merging across a gold boundary** (`t=0`, high `p`) is **penalized**; merging within a sentence
  (`t=1`) is rewarded.
- The beam selects merges by the model's **own** probabilities (it *will* wrongly merge across a
  boundary when overconfident — **no teacher forcing**); the gold-aligned loss punishes exactly
  those wrong merges. Explore-your-own-mistakes + get-slapped-for-them = the local-minima escape.

### 2.5 Training
- Fine-tune AraBERT + the scalar merge head end-to-end, **≤ 10 epochs** (it is fine-tuning, not
  from scratch).
- **Beam-3 runs during training**, not just inference.
- **Parallelization:** every candidate pair in a beam step is batched into **one** padded AraBERT
  forward pass — this is what makes beam-in-training tractable on a single GPU.
- **Windowing** for long docs; spans stay sentence-sized because merging halts at boundaries.

### 2.6 Inference
Same beam decode, no labels → surviving cuts (junctions never merged / `p < τ`) become the
predicted per-word boundaries → standard scorer (`score_cut.py`), own-best threshold, doc-macro-F1.

### 2.7 Gradient path (why it learns)
Backprop flows through `L` → the merge head → **AraBERT** (which produced every `h_[CLS]`). The
beam's discrete pick has **no gradient and needs none**: because every merge has a gold label, a
wrong merge's high `p` is pushed down by `BCE(p,0)` directly — no straight-through / RL needed.

---

## 3. Complexity & parallelization
- Per beam step: `O(B × #junctions)` candidate pairs → 1 batched AraBERT pass.
- Spans never exceed one sentence (merging stops at boundaries); windowing bounds `n`.
- Cost is dominated by repeated AraBERT passes over short span-pairs; contained by batching,
  windowing, a span-length cap, and gradient checkpointing.

---

## 4. Failure modes to expect (ranked)

**FM1 — Product-score → over-segmentation bias (most likely).**
A forest with more merges multiplies more factors `< 1` → lower product. The search therefore
*prefers fewer merges* = more boundaries. Symptom: over-segmentation (high boundary recall, low
precision). **Fix:** length-normalize the tree score (geometric mean, or divide by #merges), or
score in log-space with a per-merge bonus.

**FM2 — Degenerate collapse (merge-all / merge-none).**
Junction targets are class-imbalanced (most are `1`=within-sentence), which can push AraBERT to
output `p≈1` everywhere → merges everything → **zero boundaries** (recall 0). Or the mirror image.
Symptom: predicted boundary rate `≈0` or `≈1`. **Fix:** class-weight the BCE, monitor boundary rate
every epoch, calibrate `τ` on dev.

**FM3 — Corrupted-constituent cascade + off-distribution reads.**
Once a beam merges across a boundary, AraBERT must re-read a **two-sentence span** it almost never
saw → its downstream judgments degrade and the error cascades. Beam only helps if a *clean*
hypothesis survives; if all 3 make the same early mistake, no recovery. Symptom: errors cluster
right after one early wrong merge. **Fix:** beam diversity, cap cross-sentence span length, penalize
over-long spans.

**FM4 — Cold-start on short spans.**
Early merges feed AraBERT single words / 2-word fragments (`[CLS] the [SEP] cat [SEP]`) — off its
full-sentence pretraining distribution → noisy `p` until fine-tuned. Symptom: unstable first
epochs. **Fix:** LR warmup, a few extra epochs, careful head init.

**FM5 — Latent internal-tree instability.**
Gold constrains **where** sentences split, not the internal merge order **within** a sentence — so
the internal tree is unsupervised and arbitrary (cf. Williams et al. 2018: latent trees rarely
match syntax). Because internal merge *order* decides which spans AraBERT reads, arbitrary internal
structure injects variance into the boundary decisions. Symptom: seed-dependent, unstable trees.
**Fix:** acceptable for segmentation (only boundaries are scored), but a within-sentence structural
prior — or exact marginalization (CKY/inside-outside) instead of beam — removes the variance.

**FM6 — Beam diversity collapse.**
The top-3 hypotheses often differ by a single merge and are near-duplicates → beam degenerates to
greedy and stops exploring. Symptom: 3 beams converge to one tree. **Fix:** diverse beam search,
dedup states, small score noise.

**FM7 — Search-path dependence of supervision.**
Backprop flows only through scored probabilities; *which* candidates get scored depends on the beam
path, which depends on the (early, bad) model → a moving target. Symptom: slow/unstable early
learning. **Fix:** at every visited state, score **all** adjacent candidates (dense supervision),
not only the ones the beam selects — decouples learning from the search path.

**FM8 — Data scarcity / the coverage wall (the honest ceiling).**
174 docs, a large model, a complex objective → memorization risk; and the AraSeg wall (≈88% of
errors on unseen token-bigrams) is **not** escaped by this architecture. Symptom: train ≫ dev.
**Reality:** this model's value on closed-track is as a **different-errors voter** for the ensemble
(decorrelation), not as a standalone leaderboard jump. Measure `phi` vs the tagger — if it is not
decorrelated, it is another twin.

**FM9 — Cost / OOM.**
Beam × re-reading × retained gradients is heavy. Symptom: OOM, slow epochs. **Fix:** batching,
windowing, span cap, gradient checkpointing, shrink beam on long docs.

**FM10 — Threshold sensitivity.**
The stop threshold `τ` trades precision/recall and interacts with FM1's bias. Symptom: score swings
with `τ`. **Fix:** own-best `τ` on dev, or replace with a learned stop head.

---

## 5. Directions to enhance (researcher hand-off)
1. **Length-normalized / margin scoring** — fix FM1; add a ranking-margin term so valid merges must
   out-score invalid ones by a margin (structured objective, closer to Socher ICML-2011's max-margin).
2. **Exact structure via inside-outside / CKY** instead of beam — differentiable marginal over trees,
   removes FM5/FM6/FM7 at `O(n³)` cost (windowed).
3. **Learned stop signal** instead of a hand-tuned `τ`.
4. **Span-encoding cache** — memoize AraBERT reads of frequent spans to cut cost.
5. **Hybrid head** — add an auxiliary per-junction boundary classifier as a co-trained signal (belt
   and suspenders), then ablate to see which carries.
6. **Diverse/typed beam** — per-genre or length-typed hypotheses for the hard AraSeg genres
   (genealogy chains, legal lists, cloze).
7. **Ensemble role** — treat it as a decorrelated voter; measure and exploit `phi < 0.65` rather than
   chasing a solo win against the data wall.

---

## 6. What "working" looks like (acceptance + eval)
- **Toy acceptance test (build gate):** `the cat ate | dogs ran` → the model builds
  `(the cat ate) | (dogs ran)`, high `p` within each sentence, low `p` at `ate|dogs`, induced
  boundaries == gold. Must be able to *fail* (flip a gold label → test fails).
- **Dev metric:** doc-macro-F1 vs baseline **83.61** / gate **84.06** (+0.45 floor), own-best `τ`.
- **Decorrelation:** `phi` vs the tagger cache (pool 0.74; a genuinely different voter is `< 0.65`).
- **Sanity:** predicted boundary rate in a plausible range (~0.05–0.15), not 0 or 1 (guards FM2).

---

*Locked spec, pending Noor's revision. No build until sign-off.*
