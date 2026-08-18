# Research brief: breaking a low-resource Arabic sentence-segmentation ceiling

## What we need from you
We are stuck ~0.4 F1 short of a meaningful gain on a low-resource Arabic sentence-boundary task, and we have a precise diagnosis of *why*. We want the 2023–2026 literature (and any older classics) for **techniques that could push our precision/recall frontier OUTWARD — a genuine F1 gain, not just recalibration — under extreme data scarcity.** Specifics of the ask are in Section 7.

## 1. The task
- **AraSeg 2026** shared task (ArabicNLP @ EMNLP). **Sentence boundary detection** framed as binary token tagging: for each token position *i*, predict whether a **sentence boundary follows token *i*** (label 1) or not (label 0).
- **Metric:** per-document macro-F1 on the positive (boundary) class, averaged over documents.
- **Closed track (our constraint):** the model may be trained **only** on the **174 labeled training documents**. **No external corpora, no LM-generated text, no additional labeled data.** Pretrained encoders are allowed for initialization. (222 dev docs, 262 test.)
- **Sub-track we're optimizing:** **punctuation removed, no paragraph markers** ("NoPnx-NP") — the hardest variant. (Other tracks keep punctuation and/or paragraph breaks and score ~93–95 F1, largely saturated.)
- **Standing:** ~4th place. Top-3 teams beat us on the **same 174 documents**, so headroom demonstrably exists.

## 2. The system
- Fine-tuned **AraBERTv02** (BERT-base, Arabic) as a token classifier with a binary head, over **overlapping 180-token windows**. Weighted cross-entropy (positive class ~10% of positions; positive weight ~8). 8 epochs, batch 16, lr 5e-5.
- **Single-encoder dev macro-F1 ≈ 83.6.** (Our actual submission is a 6-encoder ensemble + semi-Markov length-prior Viterbi decoder; this brief concerns the single-encoder science, where the levers are being tested.)
- **Seed-noise floor: ±0.45 F1** (standard deviation across 3 seeds). Nothing counts as real unless it beats +0.45.

## 3. Diagnosis — what the model actually gets wrong (measured, verified)
1. **It over-segments.** False cuts outnumber missed cuts ~1.8 : 1. Precision ≈ 82, recall ≈ 88.
2. **It is confidently wrong.** Outputs are bimodal (≈0.00 or ≈1.00; almost nothing near 0.5). A "0.99 cut" is actually correct only **83%** of the time (a **+15.9-point** overconfidence gap); a "0.99 no-cut" is well-calibrated. So the miscalibration is **asymmetric — concentrated on the CUT decision.**
3. **The error is bidirectional and genre-shaped.** The model imposes a **~9–10-token "default prose sentence" length prior** (gold mean 9.88 tokens/sentence → predicted 9.04) and drags everything toward it:
   - **Over-cuts:** cloze/exercise text (1.39× too many boundaries), legal-numeric text, very long sentences, document ends (2× the false-cut rate), positions right after a bare number (2.5×), and a cloze fill-in-the-blank placeholder token.
   - **Under-cuts (at P ≈ 0.00):** genealogy "begat" chains ("X بن Y بن Z…" read as one sentence), Quranic verse openings, hadith chains of transmission (isnad), sentences that begin with a connective (و / ثم / كما), and short sentences.
4. **Coverage.** ~88–89% of errors occur on token-bigrams unseen in the 174 training docs — BUT unfamiliarity does **not** separate wrong cuts from right cuts (both are ~95% unseen; ROC-AUC of every OOV/perplexity signal ≈ 0.5). So "unfamiliar" flags *where the model fires*, not *where it fires wrongly* — an OOV-gated abstention rule is provably useless here.
5. **Gold noise is small** (~3.5% of confident false cuts look like genuine annotation error) — so the ceiling is not a labeling problem.

## 4. What we have already tried, and it failed (null = inside ±0.45)
- **Preposition/connective-targeted training** (counterfactual consistency penalty that punishes reliance on connective tokens; Mean-Teacher; Cross-View Training) — **null**.
- **Character-level augmentation** (Arabic orthographic variants: alef/hamza/taa-marbuta/ya forms, diacritics; label-preserving) — **null, and actually *increased* over-firing** across genres.
- **Token-level supervised contrastive loss** (SupCon; pull same-label token reps together) — **null**.
- **Anti-overconfidence losses:** an **asymmetric "fire-side" confidence penalty** (extra focal-style fine on confident boundary predictions at gold-negative positions only, leaving correct cuts untouched) — **null (−0.07): it raised precision ~+1.0 but lost ~1.2 recall, a 1-for-1 trade.** Symmetric entropy/confidence penalty — **null**. Label smoothing — testing.
- **Structural augmentation:** SUB2 (substructure substitution — swap a real span for another real span with an identical boundary-label pattern; in-distribution by construction) — testing; literature ceiling for label-preserving aug on a decent model is +0.5–2 F1.
- **Naive sentence recombination** — **−1.56** (went out-of-distribution).
- **Continued MLM pretraining on the 174 docs (TAPT)** — **−0.61** (overfit).
- **The ONLY method that moved up: R-Drop** (two dropout passes + symmetric-KL consistency) — **+0.33 to +0.37 F1** (precision +2.1, recall −1.5) — **but still under the ±0.45 noise floor.**

## 5. The core obstacle (our current understanding)
The model appears to sit **on its precision/recall frontier**. Anti-overconfidence and precision-oriented methods move it *along* the frontier (fire less → precision up, recall down) but do **not push the frontier outward** (no net F1). Only R-Drop pushes the frontier out at all, and only marginally. So the open problem is: **under 174 documents, how do you improve the model's actual discrimination on out-of-distribution genres/structures — pushing the P/R frontier out — rather than trading precision for recall along a fixed frontier?**

## 6. Concrete examples (real dev cases, `|` = a true boundary; P = model's boundary probability)
- **Confident over-cut (false positive):** cloze exercise — `… تكمل الجملة التالية بشكل الصحيح  |←(P=1.00, WRONG)  بدلا من فراغ ذهبت أمس …` The model slams a boundary at 100% confidence next to an exercise placeholder where there is none.
- **Confident under-cut (false negative):** genealogy chain — `… بن تارح بن ناحور  |←(P=0.001, MISSED)  بن سروج بن رعو …` A real sentence boundary sits inside a lineage; the model is 99.9% sure there is none, treating the whole chain as one sentence.
- **Confident under-cut:** Quranic verse — `… فأنزل الله تعالى  |←(P=0.002, MISSED)  لا تحرك به لسانك …`

## 7. Precise questions for you
For **binary sentence-boundary tagging on ~174 documents of morphologically-rich Arabic with punctuation removed**, where a fine-tuned BERT **over-segments confidently on out-of-distribution genres and sits on its precision/recall frontier**, find and assess:
1. **Techniques that improve GENERALIZATION to unseen structures/genres under extreme data scarcity** (not just calibration) — e.g. sharpness-aware minimization, distributionally-robust optimization / group-DRO, invariant risk minimization, meta-learning for few-shot tagging, spectral/flatness regularizers. Which have measured gains on low-resource sequence labeling?
2. **Structured / sequence-level objectives** that model boundary interdependence and directly optimize span/segmentation quality (semi-Markov CRFs, structured/minimum-risk training, pointer/boundary-ranking, sentence-length or segment-length priors baked into training). Note: a plain **linear-chain CRF is weak here** (binary labels → 2×2 transitions; boundaries can legitimately be adjacent in lists/lineages, so it tends to *worsen* the under-cut genres). What structured objective handles *variable, genre-dependent* segment lengths?
3. **Anything shown to specifically help low-resource SENTENCE / discourse / topic segmentation** (as opposed to NER/POS), especially for morphologically-rich or Semitic languages, and especially **when punctuation is removed**.
4. **How do strong systems handle genre-heterogeneous, punctuation-free segmentation** where the same model must both split dense lists AND fuse formulaic chains (genealogy/isnad/verse)? Any multi-expert, mixture, or length-conditioned approaches?
5. If possible: **what did the top systems of the AraSeg / Arabic sentence-segmentation shared tasks (2024–2026) actually do differently** on the closed track?

Please prioritize methods with **reported effect sizes on low-resource sequence labeling / segmentation**, note which are **closed-track legal** (derive only from the 174 training docs + a pretrained encoder, no new labeled or generated text), and flag anything likely to only trade precision for recall (which we've exhausted).
