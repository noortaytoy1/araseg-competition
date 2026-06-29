# Ensembling, Decoding, and the Limits of Data: A System for Arabic Sentence Segmentation at AraSeg 2026

**Team:** omar_saqr · **Tracks:** Closed + Open · **Tasks:** PA, NoPnx-PA, NP, NoPnx-NP

> Draft skeleton for the ArabicNLP 2026 (@ EMNLP 2026) shared-task system paper.
> All numbers are from EXPERIMENTS.md (per-document macro-F1; dev = selection,
> open-test = held-out verification). Fill prose where marked ⟨…⟩.

---

## Abstract ⟨150 words — write last⟩

We describe our system for the AraSeg 2026 Arabic sentence-segmentation shared
task. Framing segmentation as binary token classification, we fine-tune Arabic
encoders, ensemble them by probability averaging, and decode with a semi-Markov
dynamic program that adds a train-fit segment-length prior and forces structurally
certain boundaries. The system ranked first on all four closed-track tasks in the
development phase. Our central finding is a **negative** one: across ~20 controlled
additions — larger encoders, genre-specialist encoders, more seeds, corruption
augmentation, adversarial and label-smoothing training, and external-data
pretraining — performance **saturates**, and we show via bootstrap analysis and a
boundary-level error study that the residual error is dominated by *irreducible
punctuation ambiguity* (the comma) and *unseen-context sparsity* from a 174-document
training set, not by model capacity. We argue this makes data, not architecture,
the operative bottleneck, and we quantify exactly how much head-room remains.

---

## 1. Introduction ⟨~0.75 col⟩

- Task: predict, for each whitespace token, whether a sentence boundary follows.
  Four variants crossing paragraph-availability × punctuation-availability.
- Closed track restricts fine-tuning data to AraSeg train (174 docs); open track
  permits any public data. Metric: per-document macro P/R/F1.
- Our contributions:
  1. A strong, reproducible closed-track system (encoder ensemble + semi-Markov
     length-prior decoding) that led all four dev-phase leaderboards.
  2. A controlled saturation study: ~20 improvement attempts, each with the
     held-out test result, isolating *what does and does not help* and why.
  3. A bootstrap-based selection procedure that detects when dev is too small to
     choose between tied configurations, and a variance-minimizing tie-break.
  4. A boundary-level error analysis separating irreducible ambiguity from
     fixable sparsity, with quantified head-room.
  5. An open-track result: external boundary-recovery pretraining helps only via
     ensemble *diversity*, and only on the hardest (no-punctuation) task.

## 2. Task, Data, Metric ⟨~0.5 col⟩

- AraSeg corpus: train 174 / dev 222 / test 262 docs, ~700 tokens/doc, 8 genres.
- Tokenisation: whitespace; punctuation marks are their own tokens; paragraph
  breaks are literal `\n` tokens (PA variants); the gold boundary sits **on** the
  sentence-final token (often a punctuation mark).
- Metric: per-document binary P/R/F1 (positive = boundary), macro-averaged over
  documents — so short documents and rare genres carry equal weight.
- Cite the AraSEG dataset paper (arXiv:2606.08025).
- Note (report to organizers): official `scripts/eval.py` prints macro_recall on
  the "Precision" line and vice-versa (F1 correct); and doc-IDs are aligned across
  the four variants (NoPnx inputs are deletion subsequences of PA), a leakage
  vector we flagged rather than exploited.

## 3. System

### 3.1 Base model
- Binary token classification on a fine-tuned encoder. Label on the **last
  subword** of each whitespace token; other subwords masked (-100).
- Paragraph `\n` → a `[PAR]` special token, excluded from loss, forced to 0 at
  inference. Class-weighted cross-entropy (boundary up-weighted; rate ≈ 0.08–0.11).
- Overlapping word windows (180/stride 90 train, averaged probs at inference).
- Encoder choice (Table 1): **AraBERTv02** beats AraELECTRA and ARBERTv2 on all
  four tasks; larger/different encoders (AraBERT-large, XLM-R-large, mmBERT-base,
  CAMeLBERT-MSA/CA) all underperform the base — the data-limited signature.

### 3.2 Ensemble
- Probability averaging over 6 voters: 4 AraBERTv02 seeds + ARBERTv2 + AraELECTRA.
- Gains concentrate where punctuation is absent (Table 2): +1.1 F1 on NoPnx-NP
  from ensembling vs +0.1 on PA.

### 3.3 Decoding: semi-Markov length-prior DP
- Instead of independent thresholding, choose the boundary set maximising
  Σ log p(boundary) + Σ log(1−p) + λ·Σ log P_len(segment length), with P_len fit
  on **train** (closed-legal). Forced boundaries: the pre-`\n` token and the
  document-final token (a gold invariant in 396/396 train+dev docs).
- A two-pass *document-adaptive* variant re-estimates each document's length
  distribution from a first-pass decode and re-decodes — small but real gains on
  the no-punctuation tasks, where sentence length is short and regular.

## 4. Experiments

### Table 1 — Encoder comparison (dev macro-F1; AraBERTv02 wins 4/4)
| Encoder | PA | NoPnx-PA | NP | NoPnx-NP |
|---|---|---|---|---|
| AraBERTv02 | **94.81** | **86.43** | **92.92** | **83.07** |
| AraELECTRA | 94.43 | 85.73 | 92.18 | 82.82 |
| ARBERTv2 | 94.34 | 84.94 | 91.95 | 82.11 |
| AraBERT-large / XLM-R / mmBERT / CAMeLBERT-* | all < base (proxy) | | | |

### Table 2 — System progression (dev macro-F1)
| System | PA | NoPnx-PA | NP | NoPnx-NP |
|---|---|---|---|---|
| Rule baseline | 72.16 | 45.20 | 60.21 | 13.36 |
| AraBERTv02 (single) | 94.81 | 86.43 | 92.92 | 83.07 |
| 3-encoder ensemble | 94.94 | 87.06 | 93.16 | 84.16 |
| + DP decode (6-model mega) | 95.15 | 87.15 | 93.37 | 84.37 |
| + adaptive length prior (NoPnx) | — | 87.16 | — | 84.53 |

### Table 3 — What did NOT beat the ensemble (held-out test)
| Attempt | Result | Mechanism it lacked |
|---|---|---|
| window-240, +CAMeLBERT-MSA/CA, +mmBERT, +SaT, +aug, +soup as voters | saturated | correlated with existing voters |
| AraBERT-large, XLM-R, mmBERT, CAMeLBERT (solo) | < base 4/4 | capacity ≠ bottleneck (174 docs) |
| smooth + FGM, full pool | dev +0.2 / test −0.06,−0.07 | ensemble absorbs per-voter gains |
| greedy weighted ensemble | dev-overfit | fits dev labels |

### Negative-result vignettes (the paper's spine)
- **Length prior alone is a no-op** over a weak ensemble; it only earns weight
  (λ up to 0.4) once 6 well-calibrated voters back it — priors help only when the
  probabilities they multiply are trustworthy.
- **Per-voter improvement vanishes in-pool.** Boundary label-smoothing + FGM lift
  a *single* model by +0.68 (NoPnx-NP) but the 6-model ensemble's variance
  reduction already captures it: full-pool retraining gives +0.2 dev, −0.06 test.
  (Implication: smooth+FGM is the recipe for a *single deployable* model.)

## 5. Selection under a small dev set (bootstrap)

- Two configs tied on dev (mega+DP vs 3-encoder) differ by ≤0.1 F1; with ~28
  docs/genre, dev cannot resolve this. Bootstrap (3000× doc-resamples) gives the
  win-probability and ΔF1 CI per task (Table 4).
- Rule: trust dev only when its 95% CI excludes 0; on a tie, choose the
  **lower-variance** (fewer dev-fit knobs) config — a prior, not test-peeking.
- This flips 2/4 blind-test choices to the simpler 3-encoder ensemble; held-out
  test validates the methodology (NP +0.13; NoPnx-PA a wash, lower variance).

### Table 4 — Bootstrap (3-enc vs mega+DP)
| Task | dev P(3-enc wins) | dev 95% CI ΔF1 | decision |
|---|---|---|---|
| PA | 1.2% | [−0.41,−0.03] | mega+DP |
| NoPnx-PA | 28.5% | [−0.43,+0.26] | 3-enc (tie→simpler) |
| NP | 8.8% | [−0.53,+0.07] | 3-enc (tie→simpler) |
| NoPnx-NP | 2.0% | [−0.72,−0.02] | mega+DP+adaptive |

## 6. Error analysis: irreducible vs systematic

- **Punctuated (PA/NP):** errors concentrate on the **comma** (top FN and FP); it
  is a gold boundary only ~5–10% of the time (hadith isnad chains). Only 5–6% of
  misses sit on contexts that are usually boundaries in train; the rest are
  ambiguous or unseen → **irreducible**.
- **No-punctuation (NoPnx-*):** 88–89% of missed boundaries occur on (prev,cur)
  bigrams **never seen in 174 train docs** → sparsity, not modeling error.
- **One systematic, bounded bug:** the model over-fires on `فراغ` (fill-in-the-blank
  placeholders) — 80–98 spurious boundaries confined to 4 cloze-exercise documents
  of a genre present only twice in train. Fixing them perfectly caps at +0.54 dev;
  the principled fix is more cloze-genre data, not a dev-tuned suppression.
- Per-genre table ⟨add isnad / lyrics / genealogy / cloze breakdown⟩.

## 7. Open track: external boundary-recovery pretraining

- Method: synthesise pseudo-documents from external Arabic sentences (boundary on
  each sentence-final token), strip punctuation for the NoPnx conditions, pretrain
  the boundary classifier, then fine-tune on AraSeg train.
- Zero-shot transfer is real (0.625 proxy on dev with no AraSeg training), but a
  **single** external voter is flat-to-negative — external text supplies structure,
  not AraSeg's annotation conventions (only 174 labelled docs teach those).
- **Diversity is the lever, but it saturates fast.** Two external voters trained
  on *different* corpora (OPUS subtitles + Ashaar verse) stack to beat the closed
  ceiling on NoPnx-NP (dev 84.59>84.53, test 85.08>84.97, +0.11). A *third*
  external voter (Wikipedia, formal register) does not help (test 84.98 < 85.08):
  the diversity lever plateaus at two external voters. External data helps only
  on the no-punctuation/no-paragraph task, and only by a bounded margin.

## 8. Conclusion

- A strong ensemble+decoding system, four dev-phase leads, and — more importantly —
  a controlled demonstration that the AraSeg closed-track ceiling is set by
  *data* (174 docs, irreducible comma ambiguity, unseen-genre sparsity), not by
  architecture. We give the bootstrap tooling to select configs under a small dev
  set and the error analysis that quantifies remaining head-room.

## Reproducibility
- Env: RTX 4080, torch 2.5.1+cu121 / transformers 4.57.3 (closed); torch 2.12 env
  for `.bin` encoders. Scripts: train_encoder.py, baselines.py, predict.py,
  dp_decode.py, dp_adaptive.py, cache_probs.py, ensemble_sweep.py, augment.py,
  build_pretrain.py, open_pipeline.sh, bootstrap_stability.py, residual_errors.py.
- Every run logged in EXPERIMENTS.md (config + dev/test F1).
