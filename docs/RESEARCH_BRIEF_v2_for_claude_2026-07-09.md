# Research prompt: breaking a low-resource Arabic sentence-segmentation ceiling (updated)

You are a research assistant. Below is a fully-diagnosed low-resource NLP problem, everything we've already tried (with outcomes), and what's currently running. **We need methods we HAVEN'T tried that could push the precision/recall frontier OUTWARD (a net F1 gain), under a hard 174-document closed-track constraint.** Do not re-suggest anything in the "already tried / nulled" list. Prioritize methods with *measured* effect sizes on low-resource sequence labeling / segmentation, and flag which are closed-track-legal (derive only from the 174 train docs + a pretrained encoder; no new labeled or generated text).

## 1. Task
- **AraSeg 2026** (ArabicNLP @ EMNLP). Sentence boundary detection as **binary token tagging**: predict whether a sentence boundary follows each token. Metric: **per-document macro-F1** on the boundary (positive) class.
- **Closed track (hard constraint):** train ONLY on **174 labeled documents**. No external corpora, no LM-generated text, no extra labels. Pretrained encoders allowed for init. (222 dev, 262 test docs.)
- **4 tracks:** {punctuation kept vs removed} × {paragraph breaks kept vs removed}. Hardest = **NoPnx-NP** (no punctuation, no paragraphs). Standing: ~4th place; top-3 beat us on the *same* 174 docs.

## 2. Current REAL scores (ensemble submission, measured on dev)
| track | ensemble submission F1 | single-encoder |
|---|---|---|
| PA (punct+para) | ~94.4–95.2 | 94.4 |
| NP (punct, no-para) | ~92.9–93.4 | 92.8 |
| NoPnx-PA | ~87.3 | 86.4 |
| **NoPnx-NP** | **~85.0** | 83.1–83.8 |
Seed-noise floor: **±0.45 F1** (3-seed std). Nothing counts unless it beats +0.45.

## 3. System
Fine-tuned **AraBERTv02** token classifier, overlapping 180-token windows / stride 90, weighted cross-entropy (positive weight ~8; boundaries ~10% of tokens), 8 epochs. Submission = **6-encoder ensemble + semi-Markov DP decode**. (We A/B on a single encoder for speed, then fold winners into the ensemble.)

## 4. Diagnosis (all measured + independently verified)
1. **Over-segments:** false cuts : missed cuts ≈ 1.8 : 1. Precision ~82, recall ~88.
2. **Confidently wrong, asymmetrically:** outputs bimodal (≈0 or ≈1). A "0.99 CUT" is right only **83%** of the time (**+15.9-pt** overconfidence gap); a "0.99 NO-CUT" is well-calibrated. So miscalibration is on the **fire** side only.
3. **Bidirectional, genre-shaped, with a length prior:** the model imposes a **~9–10-token "default sentence" prior** (gold mean 9.88 tokens → predicted 9.04). It **over-cuts** cloze/exercise text, legal-numeric text, long sentences, document ends, and positions right after a bare number; it **under-cuts (at P≈0.00)** genealogy "begat" chains, Quranic verse openings, hadith isnad, connective-initial sentences, and short sentences.
4. **Coverage:** ~88–89% of errors sit on token-bigrams unseen in the 174 docs — BUT unfamiliarity does **not** separate wrong cuts from right cuts (both ~95% unseen; every OOV/perplexity signal gives ROC-AUC ≈ 0.5 for separating false-cuts from true-boundaries). So "unfamiliar" marks *where the model fires*, not *where it fires wrongly* — an OOV-gated abstention rule is provably useless.
5. **On the frontier:** precision-oriented methods slide the model *along* its precision/recall frontier (fire less → P↑, R↓) but do **not push the frontier outward** (no net F1).
6. **PA/NP "ambiguity" is mostly a comma convention, not broken gold:** measured clean fixable model errors are only **2.5% (PA) / 4.2% (NP)**; the big "defensible" bucket (~24–29%) is the model **disagreeing with gold about whether a comma (،) is a sentence boundary** — a *learnable annotation convention*, not annotator noise.

## 5. ALREADY TRIED — all nulled (inside ±0.45) or negative. DO NOT re-suggest.
- **Preposition/connective-targeted:** counterfactual-consistency penalty, Mean-Teacher (EMA), Cross-View Training → null. (Diagnostic: prepositions are only ~10% of errors.)
- **Augmentation:** character-level orthographic aug (Arabic alef/hamza/taa variants) → *worsened* over-firing; **SUB2** substructure substitution (swap real spans with identical boundary-label patterns) → null; naive sentence recombination → **−1.56** (OOD); continued-MLM pretraining (TAPT) → **−0.61** (overfit).
- **Contrastive:** token-level SupCon → null; SimCSE / sentence-paragraph contrastive → null; DAGA generation → −3.5.
- **Anti-overconfidence losses:** asymmetric fire-side confidence penalty (fine confident false-cuts only) → **−0.07** (traded P for R 1:1); symmetric entropy/confidence penalty → null; label smoothing → −0.10; soft-F1 surrogate → null.
- **Self-supervised / self-training:** BYOL on a generated corpus (closed-illegal, parked); self-training (no unlabeled pool exists in closed track).
- **OOV/perplexity "hold-fire-when-unfamiliar" gate:** dead (signal doesn't separate false-cuts from true-boundaries; AUC ≈ 0.5).
- **Decode-side:** multi-stride / center-weighted "heatmap" accumulation → +0.117 (sub-floor); genre-conditional decode thresholds → null (too few genre docs).
- **The ONLY thing that moved up:** **R-Drop** (two dropout passes + symmetric-KL) → **+0.33 to +0.37 F1**, precision-driven, reduced over-firing in *every* genre — **but still under ±0.45.** It's the one directionally-real signal.

## 6. RUNNING NOW (best-aimed remaining levers; code built + independently verified clean)
- **Minimum Risk Training (MRT)** over per-doc macro-F1 (Shen et al. 2016), warm-started from CE, **with dev-F1 early stopping** — directly optimizes the metric to push the frontier outward.
- **Filtered Semi-Markov CRF** with a **learned duration potential** (Sarawagi & Cohen 2004; Zaratiana et al. 2023) — bakes segment-length modeling into training, aimed straight at the 9-token prior. (Verified: injecting a length-10 prior reshapes its segments to ~10 tokens.)
- **SAM** + **class-asymmetric SAM** (flat-minima generalization).
- **Group-DRO** with length/perplexity pseudo-groups (worst-group robustness across genre clusters).
Results pending.

## 7. What we need from you (ranked)
1. **Methods that push the P/R frontier OUTWARD under extreme data scarcity** — not calibration (which is rank-preserving and doesn't move our F1), but techniques with *measured* gains on low-resource sequence labeling / segmentation that we have NOT tried above. (e.g. token-aware virtual adversarial training, mixup variants for tagging, structured/energy objectives, meta-learning for few-shot tagging, spectral/flatness regularizers, distributional robustness variants — but only if there's real evidence, and say what the evidence is.)
2. **The PA/NP comma-boundary lever:** how to teach the model the *annotation convention* that certain commas are sentence boundaries (and others, e.g. list commas, are not) — a disambiguation approach that gains recall without tanking precision.
3. **How strong systems handle genre-heterogeneous, punctuation-free segmentation** where one model must both split dense lists AND fuse formulaic chains (genealogy/isnad/verse) — mixture/length-conditioned/multi-expert approaches, and whether they survive n=174.
4. **What the top closed-track AraSeg / Arabic sentence-segmentation systems (2024–2026) actually did differently.**

Constraints reminder: closed-track-legal only (no new labeled/generated data); flag anything that only trades precision for recall (we've exhausted that); prefer methods with reported low-resource effect sizes over elegant-but-unvalidated ideas.
