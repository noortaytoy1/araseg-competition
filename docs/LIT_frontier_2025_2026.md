# THE FRONTIER GUIDE — "Is there anything new left" for AraSeg 2026 (closed track, 174 docs, #1 baseline)

**Repo-state recap (5 lines):** Specs on record — `SPEC_recurrent_segmenter.md`, `SPEC_delta_predictive_memory.md` (Bet B), `SPEC_delta_sat_corruption.md`, `DIRECTIVES.md` v4.1, all binding. Phases A–E, P1–P3, S1–S2 complete; 52+ tests green with `data/` hidden. Baseline = 6-model BERT ensemble (AraBERTv02/AraELECTRA/ARBERTv2/CAMeLBERT) + semi-Markov DP decode, #1 on all 4 dev tracks. Parked verdicts: TAPT/DAPT (null/hurt), generated-corpus DAPT (+0.09, kept), contrastive/morphology/surrogate-F1 (null/noise), ELECTRA/RTD (−0.02), WtP newline (tried), recursive re-reading (killed), LoRA/adapters/BitFit/PET (reviewed/parked). This is a read-only literature scout across six angles — no GPU spend, no code, no spec touched.

---

## 1. TL;DR — the genuinely new 2025-2026 strategies with any real shot

- **The closed track is very close to maxed.** Across six angle sweeps, exactly **three** methods are simultaneously (a) genuinely 2025-2026-new, (b) closed-track-legal, (c) a structural fit for per-token boundary labeling on 174 docs, and (d) low-cost/additive. Everything else is park-or-illegal. None promises more than ~+0.05 macro-F1 per member, and all three risk shrinking toward null on top of an already-averaging 6-model ensemble.
- **Noor's lead #1 (new optimizers) splits in two, and only half survives.** The *speed*-optimizer family (Muon, Sophia, Lion, SOAP/Shampoo, AdEMAMix, schedule-free) is a **confirmed dead end** — the frontier benchmark paper (Wen et al., Stanford, arXiv:2509.02046, Sep 2025) shows their famed speedups are under-tuned-AdamW artifacts that "no longer lead to downstream improvements." The *generalization*-optimizer family — **Sharpness-Aware Minimization (SAM)**, in its modern zero-cost forms — is the real answer and the single best-mechanism bet in the whole sweep.
- **Noor's lead #2 (feedback/RL fine-tuning) is worth understanding but NOT worth building.** RL-with-verifiable-reward using boundary-F1 as the reward is legal, but the closest direct precedent (Geromel & Cimiano 2024) lost **−9 to −27 F1** vs cross-entropy, the minimalist-RAFT result (arXiv:2504.11343) shows GRPO ≈ filter-then-SFT ≈ your existing training, and it re-opens your already-parked soft-F1 "noise" wall through a noisier door. Verdict: **park** — this is the honest answer to Noor's most-wanted lead.
- **The three worth trying, ranked:** (1) **SAM / flat-minima optimization** (FSAM low-data evidence "up to +15.1"; MSAM = zero overhead); (2) **DynSDPB self-distillation** (previous-mini-batch soft targets, "smaller datasets benefit more"); (3) **mono-soup tail-checkpoint averaging** (arXiv:2602.09689, one line in the training loop, ~zero downside).
- **Two adjacent, higher-integration mechanisms deserve a mention** because their *design assumptions match our regime* rather than fighting it: attention-guided weight mixup (NAACL 2024, arXiv:2403.12918) and per-document masked-LM test-time training (only if the legal/exam tracks have real distribution shift — needs a rule check first).
- **Model merging (TIES/DARE/Fisher/evolutionary) and new encoders (ModernBERT/mmBERT/AraModernBERT) are essentially all park** — merging is engineered for multi-task large-model fusion we don't have and destroys the diversity our DP decoder feeds on; new encoders are either non-Arabic or documented-worse than AraBERTv2 on Arabic token labeling.

---

## 2. RANKED TABLE

| # | Method | Year | Type | Plausible help for our token task? | Low-resource fit? | Closed-track legal? | Repo / paper | Expected value |
|---|---|---|---|---|---|---|---|---|
| 1 | **SAM (FSAM / MSAM / LightSAM)** | 2022→2026 | opt | Yes — flat minima directly targets overfit-under-scarcity | **Yes (its whole reason to exist)** | ✅ pure opt policy | [MSAM repo](https://github.com/MarlonBecker/MSAM); [FSAM 2210.05497](https://arxiv.org/abs/2210.05497); [SAM-LM 2110.08529](https://arxiv.org/abs/2110.08529) | **+0.1–0.5 F1/member, real mechanism** |
| 2 | **DynSDPB self-distillation** | 2024 | finetune | Yes — soft targets over per-token boundary logits | **Yes ("smaller data benefits more")** | ✅ own model+data | [2411.16991](https://arxiv.org/html/2411.16991) | **+0.02–0.06/member, cheap** |
| 3 | **Mono-soup (tail-checkpoint avg)** | 2026 | finetune/merge | Yes — variance reducer, flatter minima | **Yes** | ✅ own checkpoints | [2602.09689](https://arxiv.org/pdf/2602.09689) | **+0.1–0.3, ~zero downside** |
| 4 | Attention-guided weight mixup | 2024 | finetune | Yes — purpose-built low-resource overfit regularizer | **Yes (exact regime)** | ✅ base+own labels | [NAACL 2024, 2403.12918](https://arxiv.org/abs/2403.12918) | +modest, higher integration |
| 5 | Greedy soup over seeds | 2022 | merge | Marginal — within-architecture only | Yes | ✅ | [Model Soups 2203.05482](https://arxiv.org/abs/2203.05482) | +0.1–0.3, low variance |
| 6 | Per-doc masked-LM TTT (A2) | 2024-25 | test-time | Only if real train↔test shift | Narrow | ✅ *if* rule allows transductive test inputs | [TTT-AdaptNet ECCV24](https://github.com/yutianzhao-00/TTT-AdaptNet) | low/neutral, gated |
| 7 | Ensemble-consistency self-training | 2025 | data | Incremental — variant of your Bet E | Yes | ✅ unlabeled Arabic | VERIPS-style | low, incremental |
| 8 | FTFT / cartography reweighting | 2025 | data | Needs per-boundary adaptation | Partial | ✅ | [COLING 2025, 2310.06588](https://arxiv.org/abs/2310.06588) | low-mod, high variance |
| 9 | RAFT / best-of-N reranker | 2025 | feedback | Collapses to your existing SFT | No | ✅ reward=own F1 | [RAFT](https://github.com/RLHFlow/RAFT); [2504.11343](https://arxiv.org/html/2504.11343v1) | low, least-negative RL option |
| 10 | AraModernBERT as +voter | 2026 | encoder | Only if it decorrelates | Adversarial (weak on small/noisy) | ✅ init only | [NAMAA-Space/AraModernBert-Base-V1.0](https://arxiv.org/abs/2603.09982) | −0.02 to +0.03 |
| — | GRPO/RLVR F1-reward on encoder | 2024-25 | feedback | **No** — −9 to −27 F1 precedent | No | ✅ but pointless | [Geromel & Cimiano 2024](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1463164/full) | **≤0, negative** |
| — | Muon/SOAP/Sophia/Lion/AdEMAMix | 2024-25 | opt | **No** — pretraining-throughput, not generalization | No | ✅ | [Fantastic Optimizers 2509.02046](https://arxiv.org/abs/2509.02046) | **≈0/negative** |
| — | TIES/DARE/Fisher/evolutionary merge | 2023-25 | merge | **No** — multi-task/large-model tool | No | ✅ | [ICLR 2025, 2410.03617](https://arxiv.org/abs/2410.03617) | **≤0** |
| — | mmBERT / mGTE / ModernBERT / NeoBERT | 2024-25 | encoder | **No** — non-Arabic or <AraBERTv2 | No | ✅/N/A | [mmBERT](https://github.com/jhu-clsp/mmBERT) | 0 to −0.03 |
| — | TENT entropy-min / TTRL | 2021-25 | test-time | **No** — collapse on binary-imbalanced single-doc | No | ✅ inputs only | [TTRL 2504.16084](https://arxiv.org/pdf/2504.16084) | **negative (recall collapse)** |
| — | LLM-as-judge label refinement | 2024-25 | data | n/a | — | ⛔ **ILLEGAL** as supervision | [SiDyP KDD25](https://arxiv.org/abs/2505.19675) | avoid (closed) |

---

## 3. TOP 3 TO ACTUALLY TRY — with Noor's two leads given plain, concrete verdicts

### (a) Noor's lead #2 — RL / feedback fine-tuning with boundary-F1 as verifiable reward: **NOT worth building. Park it.**

This is the lead Noor most wants to work, so here is the blunt answer with the receipts. The idea — reinforcement-learn the segmenter with per-doc macro-F1 as a verifiable reward, directly optimizing the scored metric — has been tried, and the literature is discouraging on every axis that matters for us:

- **Direct precedent, unflattering:** Geromel & Cimiano (Frontiers in AI, 2024) RL'd a **BERT-base sequence labeler with F1-as-reward** and lost **−9.2 F1 on CoNLL-2003, −27.3 on OntoNotes**, winning by +1.55 only on one tiny biomedical set. Their stated mechanism: token/sequence-level F1 is too **coarse and high-variance** a signal for policy gradient when a dense per-token cross-entropy gradient already exists. That is *the same wall your parked Dice/soft-F1 surrogates hit ("noise")* — reached independently from the RL side. Two routes to "directly optimize F1," same wall.
- **The RL apparatus deflates to nothing you don't have:** the minimalist-RAFT paper (Salesforce/UIUC, arXiv:2504.11343, Apr 2025) shows **RAFT ≈ GRPO ≈ PPO**, and GRPO's edge "stems from discarding prompts with entirely incorrect responses, rather than from reward normalization." If RL reduces to best-of-N filtering, the best-of-N winner on a dense-labeled task **is the gold decode you already train on**.
- **The one on-point 2025 paper needs a regime you can't enter:** BoundRL (arXiv:2510.20151, Oct 2025) *does* make F1-reward segmentation work — but generatively, on Qwen3/Llama-3.1, with **~14,700 synthetic samples**. At 174 real docs on a closed track, that's off by two orders of magnitude and would mean discarding your #1 encoder architecture.
- **Reward-hacking risk is real:** a coarse macro-F1 reward on 174 docs invites degenerate majority-boundary-rate policies (documented failure mode, arXiv:2505.02686).

**If Noor insists on exactly one RL experiment**, the least-bad is a tightly-scoped **RAFT / best-of-N rejection-sampling reranker over the existing ensemble's semi-Markov DP candidates**, scored by per-doc macro-F1 on gold — a decode-time reranker, **not** a GRPO retrain. Rationale: RAFT captures ~all of RL's real upside (filtering) at a fraction of the instability, stays fully legal (reward = your own AraSeg F1), touches candidates you already generate, and is falsifiable in one small gated run. But my genuine recommendation is **park the whole angle before GPU** — it is a pre-registered-decision-point-shaped call and thus a **hard stop for Noor** per CLAUDE.md §4, not an auto-proceed.

### (b) Noor's lead #1 — the single best new optimizer: **SAM (Sharpness-Aware Minimization), modern form. Worth building — the best risk-adjusted bet in the sweep.**

First, kill the wrong reading of "new optimizers." The *speed* family — **Muon, Sophia, Lion, SOAP/Shampoo, AdEMAMix, schedule-free** — is a confirmed dead end for us. *Fantastic Pretraining Optimizers and Where to Find Them* (Wen, Hall, Ma, Liang; Stanford; arXiv:2509.02046, Sep 2025) shows their 1.4–2× speedups are largely under-tuned-AdamW artifacts, shrink to ~1.1× at 1.2B, and — load-bearing — **"no longer lead to downstream improvements."** Sophia specifically *overfit* (lower train loss, worse test loss) in LoRA fine-tuning — the single worst property at 174 docs. These are pretraining-throughput devices; we have a generalization problem, not a tokens/sec problem. **Do not spend a run on any of them.**

The right reading points at **SAM** — an optimization method (inner ascent step wrapping AdamW) whose entire reason to exist is **generalization under limited data**, our exact pain:
- **FSAM** (Fisher-masked SAM, EMNLP 2022 Findings, arXiv:2210.05497) — encoder fine-tuning, low-data, GLUE: "when training data is limited, FSAM improves SAM by a large margin, **up to +15.1**." Near-perfect match to closed-track AraSeg.
- **SAM-LM** (Bahri et al., ACL 2022, arXiv:2110.08529) — gains "particularly large when training data is limited," ~7.2% relative at 5% of SuperGLUE, benefits single-task settings.
- **The classic 2× cost is solved (the 2025-2026 frontier part):** **MSAM** (Momentum-SAM, NeurIPS 2025 poster, [repo](https://github.com/MarlonBecker/MSAM)) perturbs along the momentum vector already in the buffer → **zero extra forward/backward vs Adam**. **LightSAM** (arXiv:2505.24399) removes SAM's one hyperparameter (ρ). **SL-SAM** (arXiv:2602.09395, Feb 2026) activates ~21% of params, "#1 on LLM fine-tuning."

**Minimal experiment:** flag-gated optimizer swap on the existing AdamW-fine-tuned encoders. Probe with **MSAM** (zero-overhead) first; if non-negative, run **FSAM** as the primary arm (where the low-data-encoder evidence actually lives). Default off → reproduces yesterday's AdamW exactly (CLAUDE.md §2). **Pre-registered gate** (surprise-battery style): one dev track, per-token macro-F1 must be non-negative vs the AdamW member at matched steps. Honest caveats to register: headline numbers are GLUE/QA sentence-classification not token-boundary (transfer plausible, unproven); SAM *reduces* member variance, which can slightly *reduce ensemble diversity* — so measure **member-level AND ensemble** F1, not one; tune ρ or use LightSAM. **EV: +0.1–0.5 F1/member plausible, ensemble gain smaller, with a genuine mechanism.** This is a hard-stop GPU-spend decision — flag for Noor.

### (c) Third pick — DynSDPB self-distillation (the best cheap data-centric play): **worth one flag-gated arm.**

**DynSDPB** (Fu et al., 2024, arXiv:2411.16991) — no external teacher/data: the model distills from **its own previous mini-batch's soft targets**, temperature/weight adapted to running confidence. Covers encoder-only (BERT/RoBERTa/DeBERTa); authors report **"smaller datasets generally benefit more"** (RoBERTa-base RTE 60.6→68.3). Soft targets map cleanly onto per-token boundary logits — no structural obstacle. It is genuinely absent from your tested list (not TAPT, not contrastive, not a metric surrogate, not LoRA), attacks your real bottleneck (overfitting a tagger on 174 docs), and is cheap: a data-loader change + one aux KL term, default-off. **Minimal experiment:** slot the aux loss into `src/train_encoder.py` (the file that already hosts the flag-gated SupCon aux loss — plumbing precedent exists), gate on one dev track. **EV +0.02–0.06/member**, with real risk it shrinks toward null atop a 6-model ensemble that already averages variance.

*(Runner-up, if you want a fourth near-zero-cost arm: mono-soup tail-checkpoint averaging, arXiv:2602.09689 — one line in the training loop, per-encoder, keep all 6 voters + DP decode unchanged, window=1 reproduces yesterday. EV +0.1–0.3, downside ~nil.)*

---

## 4. SKIP / ILLEGAL LIST

**ILLEGAL on the closed track (flagged):**
- **LLM-as-judge / LLM-in-the-loop label refinement** (SiDyP KDD 2025 arXiv:2505.19675; LLMs-in-the-Loop ECML 2024; FreeAL) — any LLM-produced boundary labels used as supervision import external boundary knowledge, violating "boundary supervision must stay AraSeg." Even using an LLM to *clean* your 174 gold docs is a provenance-contaminating gray zone. **Hard-avoid** (fine for OPEN track only).
- **STILTs / external-labeled intermediate tasks** — already flagged closed-track-illegal in prior review.
- **Legality caveat on all test-time methods (A1/A2/A3):** legal *only* because they consume unlabeled test **inputs** transductively, never test **labels**. If AraSeg 2026's specific rules forbid *any* fitting on test inputs, these all become illegal. **STOP and confirm the rule before building any TTA arm.**

**SKIP — verified real but negative/near-zero EV (do not spend GPU):**
- **GRPO/RLVR F1-reward on the encoder** — legal but −9-to-−27 F1 precedent; re-opens parked soft-F1 wall.
- **Muon / SOAP / Sophia / Lion / AdEMAMix / schedule-free** — pretraining-throughput mirage; don't move downstream (Wen et al. 2025); Sophia actively overfits.
- **TIES / DARE / Fisher-Merging / RegMean / DoRA / evolutionary merging** — engineered for multi-task large-model fusion we don't have; cross-architecture merging is mathematically off the table (different vocabs/inits); destroys the output diversity the DP decoder feeds on.
- **ModernBERT / NeoBERT** — English-only, no Arabic. **mmBERT / mGTE** — documented below AraBERTv2 on Arabic token labeling; ModernBERT-family has a known structured-prediction penalty. **AraModernBERT** — only defensible as a decorrelation-gated +voter, not a backbone swap; weak on small/noisy data (ANERCorp 0.68, Twitter 0.49 Test-F1).
- **TENT-family entropy-min / TTRL** — collapse to majority (no-boundary) prior on binary-imbalanced single documents → recall collapse; the entire 2025 corrective literature exists because of this failure.
- **LESS / influence-based selection** — big-data pruning tool applied to a below-pruning-threshold problem (n=174); influence estimates too noisy.
- **Curriculum learning** — no compelling 2025-2026 token-labeling result; finicky, washes out with warmup.
- **Minimum-Risk Training** — not frontier; redundant with parked soft-F1.

---

## 5. FINAL HONEST VERDICT

**After this full frontier sweep: the closed track is confirmed very close to maxed, but not stone-dead.** There is no 2025-2026 method with a *strong* shot at beating the #1 baseline — the era of easy wins on 174 docs is over, and both of Noor's headline leads land on the disappointing side (RL/feedback: negative precedent and a re-opened parked wall; new optimizers: the *speed* family is a confirmed mirage). But there remains **exactly one method with a genuine, well-founded mechanism rather than a hope: Sharpness-Aware Minimization.** SAM is the only item in the entire six-angle sweep whose *published reason for existing* is generalization on limited-data encoder fine-tuning (FSAM "+15.1" in the low-data regime), whose modern variants (MSAM, LightSAM) erase its historical cost and hyperparameter, and which is a pure, additive, flag-gated optimization-policy change that is unambiguously closed-track-legal. If any single GPU run is authorized, it should be MSAM→FSAM behind a pre-registered non-negativity gate.

Below that sit two low-cost, low-downside, moderate-upside consolation bets (DynSDPB self-distillation; mono-soup checkpoint averaging) that are worth a flag-gated arm each but are honestly likely to shrink toward null on top of a 6-model ensemble that already averages away much of the variance they target.

**The blunt bottom line for Noor:** the frontier has one real mechanism left (SAM) and a couple of cheap variance-reducers. The two leads you were most excited about — feedback/RL and the fast new optimizers — are, on the honest evidence, the two you should *not* build. Every genuine remaining bet is a hard-stop GPU-spend / pre-registered-decision point under CLAUDE.md §4, so none auto-proceeds; each needs your explicit go-ahead and a ticked pre-registered reading. My ranked recommendation if you spend at all: **SAM (MSAM→FSAM) first, DynSDPB second, mono-soup third; park RL and the speed-optimizers permanently.**

**Relevant repo paths (all absolute):** `C:\Users\pc\Downloads\evolving-vlm-46\CLAUDE.md`, `C:\Users\pc\Downloads\evolving-vlm-46\DIRECTIVES.md`, `C:\Users\pc\Downloads\evolving-vlm-46\SPEC_delta_sat_corruption.md`; future SAM/DynSDPB arms slot into `C:\Users\pc\Downloads\evolving-vlm-46\src\train_encoder.py` (existing flag-gated SupCon aux-loss precedent). No files written — inline scout report per instructions.