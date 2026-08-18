# Research request: help us stack small wins into a real improvement on a low-resource Arabic NLP task

You are a research assistant. You have NO prior context — this document is self-contained. Read the background sections (A–E) in order, then complete the five steps in Section F. Cite only real published papers (venue + year), report measured effect sizes, and never recommend anything on the banned list in Section E.

---

## A. The task, from zero

**Competition:** AraSeg 2026 (a shared task at an NLP conference). **Problem:** split unpunctuated Arabic text into sentences. The model reads a document word by word and, after every word, answers one yes/no question: *"does a sentence end here?"*

**Example** (▮ marks the true sentence ends the model must find — the text has no periods to help):

> ذهب الولد إلى المدرسة ▮ عاد الأب من العمل ▮ ...
> ("The boy went to school ▮ The father returned from work ▮")

**Scoring:** F1 on the yes-answers (boundaries), computed per document, then averaged over documents ("macro-F1"). Higher is better; 100 = perfect.

**The hard constraint:** we may train on exactly **174 labeled documents**. External data, machine-generated data, and the validation set are all forbidden for training (organizer rulings — final). Pretrained models (like Arabic BERT) may be used as starting points. This makes it a pure test of *algorithms under data scarcity*.

**Terms used below** (defined once here):
- **dev set** = 222 held-out labeled documents we may *evaluate* on (never train on). All numbers below are dev macro-F1 unless marked "test".
- **noise floor ±0.45** = retraining the identical setup with a different random seed moves the score by up to ±0.45 by pure luck. Any improvement smaller than +0.45 cannot be distinguished from luck. This is why we call sub-0.45 gains "real but sub-threshold."
- **the ensemble** = our actual competition submission: 6 trained models whose per-word probabilities are averaged, plus a decoding step. Ensemble noise is smaller (≈±0.2).

## B. Where we stand

| system | score |
|---|---|
| single model (fine-tuned Arabic BERT, our workhorse for experiments) | ~83.6 dev |
| 6-model ensemble + decoder (the actual submission) | ~85.0 dev / 84.97 test |
| our rank | ~4th — the top 3 teams beat us **on the same 174 documents**, so improvement is possible |

## C. What is wrong with the model (all measured and independently verified)

1. **It memorized the training set.** On its own 174 training documents it scores **99.02**; on unseen documents **83.01**. That 16-point gap is the core disease: the model has far more capacity (135M parameters) than 174 documents can constrain.
2. **It is confidently wrong.** 98% of its answers are extreme (probability >0.9 or <0.1). When it places a *wrong* sentence break, its median stated confidence is **0.992**. Its confident "cut here" decisions are wrong 1 time in 6.
3. **Its errors are baked in, not random.** 80% of one model's errors are repeated by identically-recipe'd models trained with different random seeds. Averaging seeds cannot remove them.
4. **The errors live inside the learned representation.** We tested this three ways: (a) "is this word-context unfamiliar?" signals score AUC ≈ 0.5 (useless — everything is unfamiliar: ~95% of word-pairs in dev never appear in training); (b) a nearest-neighbor lookup over the training set's own embeddings votes the model's false cuts **UP** (0.976 — indistinguishable from real boundaries at 0.995), meaning the encoder has genuinely mapped these wrong contexts into "boundary-shaped" vectors; (c) only two weak signals see anything at the model's blind spots: prediction *instability under dropout* (AUC 0.71) and a 2D pattern described below (AUC 0.62).
5. **Genre-shaped, two-directional failure:** it cuts *too often* in exercise/quiz text and legal lists, and *refuses to cut* (confidence ≈ 0.000) inside genealogies ("X son-of Y son-of Z", where each link is its own sentence), religious chains, and sentences that start with connectives like "and/then". It drags everything toward its memorized average sentence length (~9 words).
6. **"Ensemble absorption," our meta-obstacle:** improvements to a single model repeatedly vanish when that model joins the 6-model ensemble, because all 6 voters make the *same* mistakes (error correlation 0.74). Recorded examples: a +0.9 single-model gain shrank to +0.46 in the ensemble and to a **tie** on the test set; a +0.68 gain became −0.06 at test. Lesson: single-model gains must be either very large, or come from models that make *different* mistakes.

## D. What WORKED (each explained; these are the pieces we want to stack)

1. **"Union" multi-view training, +0.9** (the biggest). The organizers provide the same 174 documents in four versions (with/without punctuation). Training one model on a document *plus its punctuated twin* (348 training items) teaches the encoder where sentences end while the answer is nearly visible (punctuation present), and part of that transfers to reading stripped text.
2. **SAM (sharpness-aware minimization), +0.39.** An optimizer wrapper that seeks flat minima in weight space — the model is penalized if a tiny nudge to its weights changes the loss a lot. Flatter minimum = less brittle memorization.
3. **R-Drop, +0.37.** Run each training example through the network twice with random dropout; penalize the two runs for disagreeing. Kills brittle, dropout-sensitive features. Measured effect: precision up ~+1.9 (fewer false cuts, in every genre).
4. **Class-asymmetric SAM, +0.32.** SAM variant that flattens the loss surface more aggressively for the "cut" class (the over-fired one).
5. **A 2D "text-as-image" refiner, +0.24.** Build a matrix where cell (i,j) = similarity of word i and word j in the encoder's embedding space; sentences appear as bright square blocks on the diagonal; a small U-Net (1M params) reads the matrix like an image and refines boundary predictions.
6. **Denser inference averaging, +0.12.** Slide the context window in smaller steps at inference and average more overlapping predictions per word, weighting central (full-context) views higher.
7. **Two weak diagnostic signals** (not F1 gains yet, but usable as features for a learned ensemble combiner): dropout-instability of a prediction (AUC 0.71 at separating the model's confident errors from its confident correct answers) and the 2D block-edge signal (AUC 0.62 at the same job).

Note the levers sit at different levels — data (1), optimizer (2, 4), loss (3), inference (5, 6) — which is why we believe they may stack.

## E. What FAILED here (banned — do not re-suggest; each was measured to the floor described in A)

Data augmentation: character-level spelling noise (made over-cutting worse), span-swapping, sentence recombination (−1.56), continued masked-LM pretraining on the 174 docs (−0.61), synthetic-data generation (illegal anyway). Contrastive: token-level supervised contrastive, SimCSE-style, paragraph-level, sentence-order objectives — all null. Anti-overconfidence losses: penalizing confident false cuts, entropy penalties, label smoothing, differentiable-F1 — all null (they trade precision for recall 1:1; the frontier doesn't move). Confidence gates: "abstain when unfamiliar" (AUC 0.5), nearest-neighbor voting (endorses the errors), hard filtering by dropout-instability (kills more true boundaries than false cuts at every threshold — usable only as a feature). Architectures: bigger encoders (Arabic-BERT-large, XLM-R-large — both LOSE to base on 174 docs), an end-to-end semi-Markov CRF (−4.1: its from-scratch components starved on 174 docs; a repaired version is queued), reinforcement/minimum-risk training (heavy negative), group-robust optimization (−0.92), audio/TTS expert fusion (the learned fusion gate weighted the audio expert 0.04 — the model ignored it; unpunctuated *written* text carries no recoverable prosody for a TTS encoder).

---

## F. YOUR JOB — five steps, in order

**Step 1 — Composition.** We are about to train ONE recipe combining levers D1+D2+D3 (union data + SAM + R-Drop). Search the literature for evidence on whether weight-space flatness methods (SAM) and output-consistency methods (R-Drop) **compose or overlap** when combined. Deliverable: a verdict ("expect additive / expect overlap") with citations, plus the ONE additional regularizer with the best published evidence of composing with both (candidates to check: stochastic weight averaging, EMA, mixup, stochastic depth).

**Step 2 — Partial fine-tuning.** Given the 16-point memorization gap (C1), find the best-evidenced **parameter-efficient or partial fine-tuning recipe** for token-level tagging with ~200 training documents: freezing bottom layers, LoRA (low-rank adapters), BitFit (bias-only), Child-Tuning (gradient masking), layer-wise learning-rate decay. Deliverable: ONE concrete recommended configuration (which layers/ranks/learning rates) with published low-resource evidence that it *reduces the train-dev gap*, not merely matches full fine-tuning.

**Step 3 — Surviving the ensemble.** Given the absorption problem (C6), find evidence on: (a) whether retraining ALL ensemble members with one improved recipe re-correlates their errors and forfeits the gain; (b) ensemble combination methods beyond probability averaging for sequence labeling — especially a trained meta-combiner ("stacking") over out-of-fold member predictions plus auxiliary features (we have two: D7). Deliverable: the combination approach with the best published evidence for small-data sequence labeling, with its measured gain over averaging.

**Step 4 — Moving a poisoned representation.** Given C4 (the errors are embedded in representation space), critique our queued fix: contrastive training where anchors = true-boundary contexts and hard negatives = the model's own highest-confidence wrong-cut positions mined from training data, with a margin loss. Specifically: known failure modes when the hard-negative miner IS the model being fixed, and any better-evidenced technique for surgically relocating a specific error cluster in representation space at this data scale. Deliverable: "run as designed" / "run with these changes" / "replace with X", with citations.

**Step 5 — The blind spot check.** After steps 1–4, name the ONE technique with real published evidence for exactly this regime (token tagging, <500 training docs, pretrained encoder, memorization gap) that we have neither tried (Section E) nor queued (Sections D/F). If nothing qualifies, say so explicitly — a confirmed "your coverage is complete" is a valid and valuable answer.

**Response format:** number your answers Step 1–5. Every claim carries a citation (venue+year) and a measured effect size where one exists. Flag speculation as speculation. Do not exceed ~2 pages per step.
