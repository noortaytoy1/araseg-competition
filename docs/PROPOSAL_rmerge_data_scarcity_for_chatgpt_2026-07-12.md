# Recursive merge segmenter + the data-scarcity wall — brief for external brainstorming (ChatGPT)

*Self-contained; the reader has no repo access. Goal: enhance the proposal below to beat a data-scarcity
ceiling on Arabic sentence segmentation. 2026-07-12.*

---

## 1. The task

**AraSeg 2026** (ArabicNLP shared task), **closed track, NoPnx-NP** = *No Punctuation, No Paragraph
marks*. Input: a raw sequence of Arabic **words** with punctuation and paragraph breaks stripped out.
Output: for **each word**, a binary label — does a sentence end after this word? Scored by **boundary
precision / recall / F1**, aggregated as **document-macro-F1**.

- Baseline **83.61**, acceptance gate **84.06**, **stretch goal 85.5** (dev macro-F1).
- **Closed track rules**: only the provided training set may be used to fit parameters. **No external
  labeled data. No training on dev.** A single Arabic encoder (AraBERTv02) is the backbone.

### Data facts (measured)
- **174 training documents**, 222 dev documents.
- Documents are long: **median 426 words**, p90 ~900, one doc 15,472 words.
- Sentences are short: **median 8 subword-tokens**, p99 ~50.
- **~10% of word-junctions are true sentence boundaries** (class imbalance within-sentence:boundary
  ≈ **8.8 : 1**).

### The wall (why this is hard)
Error analysis shows **~88% of dev boundary errors occur at token-bigrams that never appear in the 174
training documents.** This is a **coverage / data-scarcity wall**: the model fails on unfamiliar word
pairs it simply never saw. More capacity or a cleverer objective does not, by itself, cross it — the
missing ingredient is *coverage of Arabic word-adjacency patterns*.

---

## 2. The proposal (our current model)

Reframe segmentation as **recursive sentence construction**, not per-token tagging.

**The scorer.** One shared AraBERTv02 reads two adjacent text pieces and outputs the probability they
belong to the *same sentence*:
```
input  =  [CLS] tokens(A) [SEP] tokens(B) [SEP]
p      =  sigmoid( w · h_[CLS] + b )   ∈ (0,1)  =  P(A, B are the same sentence)
```
The same weights score every merge at every level of the recursion — it **re-reads the real tokens**
each time (no pooled/stored vectors, no separate composer).

**Training — per-pass recursive self-gluing (no teacher forcing).**
Start with every word as its own piece. Repeat *passes*:
1. Score every adjacent piece-pair (one batched forward); apply **class-weighted binary cross-entropy**
   against gold — within-sentence pair → target 1 (should glue), cross-boundary pair → target 0 (must
   **not** glue; up-weighted 8.8×, punished).
2. The model then **glues its own most-confident pairs first** (highest P above a threshold, taken
   non-overlapping left-to-right after that) — its *own* decisions, not the answer key. A span-length
   cap forbids over-long merges.
3. The glued pieces feed the next pass (**the recursion**), *including any wrong cross-boundary lumps*,
   so the model is trained on the corrupted states it actually produces and learns to place the
   remaining boundaries after a mistake. Stop when a pass glues nothing.
- **Cold-start safety**: the merge head is initialized biased toward *gluing nothing* (P(glue)≈0.12), so
  a random model fails toward under-gluing (recoverable) rather than fusing away real boundaries
  (irreversible). It then learns its way up.

**Inference — beam search (width 3).** Same scorer; explore a few alternative gluings and keep the
highest-scoring whole segmentation (length-normalized geometric-mean score, own-best stop threshold
tuned on dev). Surviving un-glued junctions = predicted sentence boundaries.

**Why this shape:** boundaries become a by-product of a learned compositional process. The hope is that
"does this Arabic word-span compose with the next?" generalizes to unseen word pairs better than a flat
per-junction classifier — i.e., that composition is a *smoother* signal than memorized bigrams.

---

## 3. What is already ruled out (do not re-propose)

- **Diverse-encoder ensembles** (AraBERT + 4 CAMeLBERT variants, corpus-diverse pools): tested,
  **audit-clean NULL** — the variants were individually too weak and added no decorrelated signal. The
  AraBERT-core pool at 84.94 stayed best. (This recursive model is a *single-encoder* effort; ensembling
  is not the lever here.)
- **Training on dev** — forbidden (closed track).
- **External labeled corpora / other encoders** — out of scope for the closed track.
- Heavy machinery constraints: no new dependencies, no CRF, no reinforcement-learning frameworks; the
  encoder is fixed as AraBERTv02.

## 4. Directions already on the table (context, not yet decisive)
- **Character-level augmentation** flagged in the low-resource literature as the strongest, safest lever
  for morphologically rich Arabic — label-preserving perturbations that attack the unseen-bigram
  coverage gap directly.
- **Self-training / self-distillation** on a large *unlabeled* Arabic corpus (e.g. BYOL-style), using the
  model's own confident segmentations as targets — needs a clean, non-leaking implementation.
- **LM-generated pseudo-labeled sentences** — pending a ruling on whether generated data is admissible
  under the closed track.
- **Retrieval over generation** for domain text (e.g. Hadith chains): retrieve real sentence-segmented
  spans rather than hallucinate them.

---

## 5. The ask for ChatGPT

**Given the closed-track constraint (only the 174 training docs may fit parameters; one AraBERTv02
encoder) and the finding that ~88% of errors are on unseen word-bigrams — how would you enhance the
recursive merge proposal in §2 to cross the data-scarcity wall and reach dev macro-F1 ≥ 85.5?**

Concretely, we want ideas on:
1. **Data multiplication that is admissible closed-track** — label-preserving augmentation (character,
   subword, word-swap, span-reordering, sentence-shuffling to synthesize new *cross-sentence* boundary
   contexts), and how to augment for a *pairwise merge* objective specifically (what perturbation
   preserves the merge/boundary label of a junction?).
2. **Making the merge scorer generalize to unseen bigrams** — auxiliary self-supervised objectives on
   the merge representation, consistency/contrastive losses over augmentations, morphology-aware
   features, or a smoothing prior that reduces reliance on memorized adjacency.
3. **Self-training on unlabeled Arabic** without leakage — confidence filtering, agreement between the
   per-pass and beam decoders as a pseudo-label gate, iteration schedule, collapse avoidance.
4. **Regularization for 174 documents** — over a recursive objective that visits many self-generated
   states (high effective sample count but low document diversity): what prevents memorization?
5. **Anything structural** in the per-pass recursion or beam that would specifically help the hard,
   unfamiliar junctions rather than the easy ones.

Please prioritize ideas by expected payoff-to-effort under the closed-track constraint, flag any that
would violate "no external data / no dev-training," and be concrete enough to implement.
