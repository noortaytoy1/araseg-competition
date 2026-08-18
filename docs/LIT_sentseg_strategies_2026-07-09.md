I have all six angle reviews synthesized and cross-checked against the repo's tested-list and closed-track rules. The task is pure synthesis into the requested markdown format. No new research or file inspection is needed. Let me produce the deliverable.

# Pretrain / Finetune / Augmentation guide for AraSeg 2026 — 174-doc closed track

## 1. TL;DR — highest-value NEW strategies (not already tested here)

- **Augmentation (strongest signal in the whole review):** In-training **punctuation/cue-dropout corruption on the supervised AraSeg objective** — the SaT recipe (drop/duplicate/strip Arabic punctuation ".،؛?!"، tatweel/diacritics on a fraction of batches). SaT's own ablation shows removing this collapses no-punctuation F1 from **54.4 → 10.2** — direct, quantified, segmentation-specific evidence. You tested WtP as *pretraining*, never as an in-training supervised corruption schedule. This targets exactly your NoPnx headroom. **Top pick overall.**
- **Finetune (best evidence-to-cost, matched to the scored metric):** **Boundary decision-threshold calibration to per-document macro-F1, out-of-fold and per-track**, paired with **boundary smoothing** (Zhu & Li, ACL 2022) as a pre-threshold calibration regularizer. AraSeg scores exact, tolerance-free per-doc boundary-F1 whose Bayes-optimal threshold is provably not 0.5. Task #20's "OOF threshold" is the seed; per-track thresholds + boundary smoothing are the new parts.
- **Finetune (cheapest stability lever):** **Top-layer re-init + bias-corrected Adam + train-longer** (Zhang et al., ICLR 2021, `asappresearch/revisit-bert-finetuning`). Canonical few-sample fix; injects seed diversity that *helps* your 6-model soup. One-line change, closed-track-clean.
- **Pretrain (only new objective with a direct SBD number):** **Punctuation-restoration pretraining** on a large external unlabeled Arabic corpus, then fine-tune the head on the 174 docs (Min et al., NAACL/RepL4NLP 2025, arXiv:2402.08382) — .81→.98 SBD F1 on PTB, almost pure **recall**, your NoPnx weakness. Mechanistically distinct from the killed TAPT (different objective, external corpus, not continued-MLM on 174 docs).
- **Pretrain done-right (steelman of your dead TAPT):** if you run the live DAPT at all, run it under **RecAdam weight-anchoring** (Chen et al., EMNLP 2020, `Sanyuan-Chen/RecAdam`) with early-stop ≤2–3 epochs + lower layers frozen — AdaptSum shows exactly your "worse after ~3 epochs on a small corpus" collapse and its fix. Data-free, closed-track-clean.
- **Augmentation runner-up:** **Substructure Substitution (Sub²)** — whole-sentence content-swap at fixed boundary structure (Shi/Livescu/Gimpel, Findings ACL 2021). The *correct* "cut-and-splice" that fixes the desync that killed your random-recombination (−1.56); its documented sweet spot is small seed sets.

## 2. Ranked table

| # | Strategy | Bucket | Evidence it helps SEGMENTATION | Low-resource? | Tried here? | Closed-track-legal? | Repo | Expected value for us |
|---|---|---|---|---|---|---|---|---|
| 1 | In-training punctuation/cue-dropout corruption (SaT recipe) | aug | **Direct**: SaT 54.4→10.2 no-punct F1 ablation (EMNLP 2024) | Yes (robustness/low-punct) | No (WtP tested as *pretraining*, not this) | **Yes** — corrupts AraSeg's own text | `segment-any-text/wtpsplit` | **High** on NoPnx tracks; low cost |
| 2 | OOF + per-track threshold calibration + boundary smoothing | finetune | Indirect-but-matched: exact per-doc macro-F1, optimal thr ≠ 0.5; Zhu & Li ACL 2022 calibration | Yes (post-hoc) | Partial (#20 global OOF only) | **Yes** | none needed (in-repo) | **Medium-high**, near-zero GPU |
| 3 | Top-layer re-init + debias-Adam + train-longer | finetune | Mechanism (top layers bad init); GLUE only, not token | **Yes (canonical few-sample)** | No | **Yes** | `asappresearch/revisit-bert-finetuning` | **Medium**, cheapest; ensemble-synergistic |
| 4 | Punctuation-restoration pretraining (external Arabic corpus) | pretrain | **Direct SBD .81→.98 F1** (PTB, T5); recall-heavy | No curve; Alam W-NUT'20 tunes objective LR | No (≠ TAPT) | **Yes** (mask punct on raw Arabic) | `xashru/punctuation-restoration` (objective); AraPunc (Arabic proof) | Medium; risk = BERT not recall-starved |
| 5 | Sub² same-label / whole-sentence substitution | aug | Inferred (POS only in-paper, not boundary) | **Yes, strongest when data small** | No (≠ shuffle/recombination) | **Yes** | none (impl ~50 lines) | Medium |
| 6 | RecAdam-anchored DAPT (early-stop, freeze lower layers) | pretrain | Mechanism (fixes overfit collapse); GLUE/summ, not token | Yes (AdaptSum, smaller corpus = more forgetting) | No (steelman of dead TAPT) | **Yes** (data-free) | `Sanyuan-Chen/RecAdam` | Medium; coin-flip-positive probe |
| 7 | SaT+LoRA adapted on 174 AraSeg docs as 7th voter | pretrain/transfer | Yes (ar 79.9→86.3 F1 w/ LoRA) | **Strong (10–100 sents)** | No (#19 was open-track off-the-shelf) | **Yes** (LoRA data = your labels) | `segment-any-text/wtpsplit` (MIT) | Low-medium; XLM-R voter may be redundant |
| 8 | Cutoff / HiddenCut + per-token JS-consistency | aug | No (GLUE/MT only); mechanism preserves labels | No | No (≠ SimCSE/SupCon) | **Yes** | `dinghanshen/Cutoff`; HiddenCut ACL'21 | Low-medium; only after 1&5 |
| 9 | Attention-guided weight mixup (learnable Mixout/RecAdam) | finetune | No (GLUE only); token-task coverage unverified | **Yes (300–500 ex.)** | No | **Yes** | none confirmed | Low-medium; fallback to #3 |
| 10 | Mixout / SWA / model-soups | finetune | No (classification/QA); soup can't mix diff-vocab encoders | Partial | No | **Yes** | `wtpsplit`-unrelated; ICLR/ICML repos | Low; redundant with your DP ensemble |
| 11 | LoRA/BitFit/Child-Tuning as overfit control | finetune | Underperforms full-FT on token labeling | Plausible unproven | No | **Yes** | LoRA/BitFit official | Low for accuracy |
| — | Full semi-CRF / pointer decoder (SegBot) | finetune | CRF marginal on strong encoder; SegBot 92.2 F1 but data-hungry | No | You already run semi-Markov DP | **Yes** but "No CRF" scope-barred / arch change | `urchade/Filtered-Semi-Markov-CRF`; SegBot | Skip |
| — | Dice / focal / self-adjusting DSC | finetune | Dice +0.29, focal +0.06 F1 (NER) — sub-noise | — | **Yes (null)** | Yes | — | Skip (relitigation) |
| — | TSSP / SegNSP / StructBERT / SOP | pretrain | Topic/inter-sentence, wrong granularity | No | ~coherence-null family | off-task | — | Skip |
| — | SeqMix token-label mixup | aug | **Breaks alignment (manifold intrusion)** | AL-loop only | ~recombination-null family | needs AL loop | `rz-zhang/SeqMix` | Skip |
| — | Back-translation + boundary projection; MSLM; cross-lingual boundary head | aug/pretrain/transfer | Fragile projection / wrong granularity (morpheme) / needs external labels | — | ~generate-relabel-null | **No** (open-track or morpheme) | — | Skip |

## 3. Top 3 to actually try next

**1. In-training punctuation/cue-dropout corruption (SaT recipe).**
- Paper/repo: Frohmann et al., *Segment Any Text*, EMNLP 2024, arXiv:2406.16678 · `github.com/segment-any-text/wtpsplit` (recipe reference; you implement the schedule in your own dataloader, not import SaT weights — keeps it closed-track).
- Minimal experiment: add `--cue-dropout-prob` to the existing dataloader. On a fraction of training batches (start 10–15%), stochastically drop/duplicate Arabic punctuation ".،؛?!"، and strip tatweel/diacritics; boundary-after-token labels are recomputed trivially (removing a comma doesn't move a boundary). Train the same encoders, read the **NoPnx-track recall/F1 delta** on frozen folds vs no-corruption. Keep iff NoPnx lifts without punctuated-track regression. **Hard-stop note:** this touches `SPEC_delta_sat_corruption.md` — a pre-registered decision point; confirm before GPU spend.

**2. OOF + per-track boundary-threshold calibration + boundary smoothing.**
- Paper: Zhu & Li, *Boundary Smoothing for NER*, ACL 2022, arXiv:2204.12031 (finetuning-time calibration regularizer); F1-threshold theory: Fan & Lin, `csie.ntu.edu.tw/~cjlin/papers/threshold.pdf`. No new dependency.
- Minimal experiment: (a) sweep the DP decode's boundary decision threshold on out-of-fold predictions to maximize **per-document macro-F1**, once globally and once **per track**; use nested CV, keep single-global-threshold as the pre-registered safe fallback (per-track risks fold overfit at 174 docs). (b) Optionally add boundary smoothing as a finetuning-time regularizer so pre-threshold probs are better calibrated. Near-zero GPU; composes with the existing ensemble.

**3. Top-layer re-init + bias-corrected Adam + train-longer.**
- Paper/repo: Zhang et al., *Revisiting Few-sample BERT Fine-tuning*, ICLR 2021, arXiv:2006.05987 · `github.com/asappresearch/revisit-bert-finetuning`.
- Minimal experiment: re-initialize the top 1–3 transformer layers of each encoder, enable bias-corrected Adam, extend training iterations. Read dev F1 + seed-variance per fold. Its own second finding (fancier regularizers shrink once debias-Adam + train-longer are fixed) tells you to test this *one* thing and rationally park Mixout/R3F/soups.

## 4. Skip these (appealing but the evidence — or your own results — says no)

- **Full semi-CRF training / pointer-net decoder (SegBot):** you already run semi-Markov DP; CRF is marginal on a strong encoder and scope-barred ("No CRF"); SegBot is a data-hungry architecture change, not additive.
- **Dice / focal / self-adjusting DSC:** literature's own numbers (Dice +0.29, focal +0.06 F1 on NER) explain your "within noise" — relitigation.
- **TSSP / SegNSP / StructBERT sentence-objective / SOP:** topic/inter-sentence coherence, wrong granularity for boundary-after-token; same family as your null coherence/shuffle arm.
- **SeqMix (and back-translation + boundary projection, generate-then-relabel):** break token-label alignment (manifold intrusion / fragile projection) — the exact failure that sank random-recombination (−1.56) and DAGA (−3.5).
- **MSLM multilingual segmentation transfer:** wrong granularity (word/morpheme, not sentence) — a title trap.
- **Cross-lingual boundary-head transfer (mBERT/XLM-R zero/few-shot), punctuation-restoration cross-lingual transfer:** require external source-language boundary supervision → open-track only; distant-language ceiling for Arabic; no SBD-specific evidence.
- **Ersatz:** punctuation-anchored only — structurally cannot help on NoPnx, where your headroom is; dominated by SaT.
- **Model-soups / LoRA-DAPT / BitFit-for-accuracy:** you already ensemble in output space (can't soup different-vocab encoders); PEFT underperforms full-FT on token labeling and you have no parameter budget problem.
- **Continued-MLM TAPT on the 174 docs, plain:** already hurt; only defensible as the RecAdam-anchored external-corpus variant (#6), which is a different intervention.

## 5. Can any of this beat #1, or are we confirming the ceiling?

Mostly confirming the ceiling — but there is **one plausible real lever**: in-training punctuation/cue-dropout corruption has direct, quantified, segmentation-specific evidence on exactly your NoPnx weakness (SaT 54.4→10.2), so it has a genuine shot at moving the punctuation-poor tracks; everything else is a cheap variance-reducer or coin-flip probe that, against a #1-on-all-4-tracks ensemble, most likely lands within seed noise.