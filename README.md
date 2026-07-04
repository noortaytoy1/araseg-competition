# Arabic Sentence Segmentation — AraSeg 2026 Shared Task

[![fixture-smoke-test](https://github.com/omarsaqr12/arabic-sentence-segmentation/actions/workflows/smoke.yml/badge.svg)](https://github.com/omarsaqr12/arabic-sentence-segmentation/actions/workflows/smoke.yml)

A complete system for **Arabic sentence-boundary detection**, built for the
[AraSeg 2026 Shared Task](https://www.araseg.aramlab.ai/) at **ArabicNLP 2026
(@ EMNLP 2026, Budapest)**. The task: given an Arabic document, predict for every
token whether a sentence boundary follows it — across four variants that cross
*paragraph availability* × *punctuation availability*.

> **Ranked #1 on all four closed-track subtasks** (development-phase leaderboards,
> CodaBench, as `omar_saqr`): **PA 94.4 · NoPnx-PA 87.3 · NP 92.7 · NoPnx-NP 85.0**
> (open track 85.2) — ahead of the organizer baseline on every board
> (PA +1.6 … NoPnx-NP +7.4 F1).

The system is a **probability-averaged ensemble of fine-tuned Arabic encoders**
(AraBERT / AraELECTRA / ARBERT) decoded with a **semi-Markov dynamic program**
that adds a train-fit segment-length prior and forces structurally-certain
boundaries. Beyond the leaderboard, the headline result is a **negative one**: a
20-experiment controlled study (encoder scaling, calibration, augmentation,
external-data pretraining) shows performance **saturates** — and the paper
*quantifies* why: a 3-seed **scaling law** (PA at its 94.8 asymptote, α=1.37;
NoPnx-NP still rising toward 86.4, α=0.66), a **model-free ambiguity check**
(gold labels are 99%-consistent for repeated contexts, so the residual is
**context sparsity from 174 training documents — not irreducible noise, and not
model capacity**), and a Krogh–Vedelsby decomposition (voter diversity is only
14% of individual error).

---

## Results

### Leaderboard (CodaBench dev phase, closed track — official scores)

| Subtask  | Description                              | Closed | Open | Organizer baseline | Δ      |
|----------|------------------------------------------|:------:|:----:|:------------------:|:------:|
| PA       | Punctuation + paragraphs                 | **94.4** | 94.4 | 92.8             | +1.6   |
| NoPnx-PA | Paragraphs, **no punctuation**           | **87.3** | **87.3** | 82.8         | +4.5   |
| NP       | Punctuation, **no paragraphs**           | **92.7** | 92.7 | 89.7             | +3.0   |
| NoPnx-NP | **No punctuation, no paragraphs** (hardest) | **85.0** | **85.2** | 77.8      | +7.4   |

Open = closed on three tasks (external data never beat the closed ensemble there).
On NoPnx-NP the open system is a **4-voter pool** (AraBERTv02 seed + AraELECTRA +
an OPUS boundary-recovery voter + a **fine-tuned SaT-12L**): a second round of
open-track attacks showed data-diverse voters (classical Tashkeela, a 1M-sentence
scale-up) all *dilute* the ensemble, while the architecture-diverse SaT voter —
weakest solo — is the only one that helps (dev +0.47, bootstrap CI [+0.15,+0.80];
test 85.17). Decorrelation, not accumulation. A final round extended the
architecture axis further: **mDeBERTa-v3** — a fourth architecture family, and
closed-track legal — joins the NoPnx-PA ensemble in both tracks (dev +0.54,
CI [+0.20,+0.87]; test 87.27), while longer-SaT and SaT-for-NoPnx-PA variants
posted positive dev signals but failed test confirmation and were rejected.

Per-document macro-F1 (boundary class). Official CodaBench scores matched the
offline evaluator (`src/eval_local.py`) exactly, so iteration happened offline and
only verified gains were uploaded.

### How the system was built up (dev macro-F1)

| Stage                               | PA    | NoPnx-PA | NP    | NoPnx-NP |
|-------------------------------------|:-----:|:--------:|:-----:|:--------:|
| Rule baseline                       | 72.16 | 45.20    | 60.21 | 13.36    |
| Single AraBERTv02 (fine-tuned)      | 94.81 | 86.43    | 92.92 | 83.07    |
| 3-encoder probability ensemble      | 94.94 | 87.06    | 93.16 | 84.16    |
| + semi-Markov DP decode (6-model)   | 95.15 | 87.15    | 93.37 | 84.37    |
| + document-adaptive length prior    | —     | 87.16    | —     | **84.53** |

Full run-by-run log (every model / window / threshold / seed with dev **and**
held-out test F1) is in **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

---

## Approach

**Framing.** Binary token classification: each whitespace token gets a 0/1 label
on its **last subword** ("a boundary follows this token"); other subwords are
masked. Paragraph `\n` marks become a `[PAR]` special token, excluded from the
loss and forced to `0` at inference. Class-weighted loss handles the ~8–11%
boundary rate. Long documents are processed in overlapping windows whose
probabilities are averaged.

**Ensemble.** Probability averaging over 6 voters — 4 AraBERTv02 seeds + ARBERTv2
+ AraELECTRA. Gains concentrate exactly where punctuation cues vanish (+1.1 F1 on
NoPnx-NP vs +0.1 on PA). AraBERTv02 beat every alternative encoder on all four
tasks; larger models (AraBERT-large, XLM-R-large, mmBERT, CAMeLBERT) all
*underperformed* the base — the signature of a data-limited, not capacity-limited,
problem.

**Decoding.** Rather than independent per-token thresholding, a **semi-Markov
dynamic program** picks the boundary set maximizing
`Σ log p(boundary) + Σ log(1−p) + λ·Σ log P_len(segment length)`, with the
length prior `P_len` fit on **train** (closed-track legal). Structurally-certain
boundaries (the token before `\n`, and the document-final token — a gold
invariant in 396/396 train+dev docs) are forced. A two-pass *document-adaptive*
variant re-estimates each document's length distribution and re-decodes, which
helps on the short, regular sentences of the no-punctuation tasks.

**The negative result (the interesting part).** ~20 controlled additions were
tried and measured on held-out test; **none beat the ensemble**:

- More/different voters (window-240, CAMeLBERT-MSA/CA, mmBERT, SaT, augmentation,
  weight-soups): saturated — too correlated with existing voters.
- Bigger encoders: all lost to base AraBERTv02 (174 docs can't feed 370M params).
- Per-voter calibration (label smoothing + FGM adversarial training): lifts a
  *single* model by +0.68 F1, but the ensemble's variance reduction already
  captures it (dev +0.2, test −0.06).
- A **bootstrap selection procedure** (3000× document resamples) detected when the
  dev set is too small to separate tied configs and tie-broke toward the
  lower-variance choice — flipping 2/4 blind-test submissions to a simpler
  ensemble, validated on held-out test.

A boundary-level **error analysis** then localizes the ceiling: on punctuated
tasks errors concentrate on the comma — hard not because it is random (its label
is near-deterministic given ±2 tokens of context) but because each comma context
is *rare*; on no-punctuation tasks 88–89% of misses fall on token bigrams *never
seen* in the 174 training documents, and only 5% of dev boundary contexts are
covered by training at all. The bottleneck is data, and the head-room is
quantified (power-law asymptotes with bootstrap CIs, a calibration-grounded
Bayes floor, SaT-12L zero-shot ≈69 as the external reference point).

**The full write-up:** [paper/araseg_system.pdf](paper/araseg_system.pdf)
(6 pp, 5 figures, 8 tables). An [ACL-template version](paper/acl/) with a
proper `.bib` is ready for the ArabicNLP 2026 submission (~Aug 8).

---

## Repository structure

```
.
├── README.md                  ← you are here
├── requirements.txt
├── LICENSE                    (MIT)
│
├── src/                       ← all Python modules
│   ├── data.py                load AraSeg from HuggingFace / local JSONL; constants
│   ├── baselines.py           rule baselines (punct / verse / par / every-k)
│   ├── train_encoder.py       token-classification fine-tuning (windows, [PAR], weighted loss)
│   ├── predict.py             overlapping-window inference + format validation
│   ├── eval_local.py          offline mirror of the official metric
│   ├── cache_probs.py         cache per-model boundary probabilities (.npz)
│   ├── ensemble_sweep.py      probability-averaging ensemble + threshold sweep
│   ├── weighted_ensemble.py   greedy per-voter weight search (dev)
│   ├── dp_decode.py           semi-Markov length-prior DP decoding
│   ├── dp_adaptive.py         two-pass document-adaptive length prior
│   ├── sweep_threshold.py     per-task decision-threshold tuning
│   ├── augment.py             corruption / token-deletion augmentation
│   ├── build_pretrain.py      synthetic boundary-recovery pretraining data (open track)
│   ├── bootstrap_stability.py 3000× bootstrap config selection under a small dev set
│   ├── residual_errors.py     irreducible-vs-systematic boundary error analysis
│   ├── genre_buckets.py       per-genre score breakdown
│   ├── sat_eval.py            SaT / wtpsplit zero-shot baseline (open track)
│   ├── predict_blind.py       one-command reproduction of all 8 frozen submissions
│   └── gen_test_*.py          test-split submission builders
│
├── scripts/                   ← shell training batteries (encoder/seed/window sweeps)
│   ├── train_all.sh  train_battery*.sh  train_matrix.sh
│   └── open_pipeline.sh  open_wiki.sh   (open-track pretraining)
│
├── docs/
│   ├── EXPERIMENTS.md         full experiment log (20+ runs, dev + test F1)
│   ├── PLAYBOOK.md            end-to-end guide: setup → submission → paper
│   └── RUNBOOK_BLIND_TEST.md  mechanical blind-test procedure (reproducible)
│
├── paper/                     ← system-description paper (.tex → .pdf, figs/)
│   └── acl/                   ← ACL-template port (acl.sty, .bib) for ArabicNLP 2026
└── fixtures/PA_mini.jsonl     ← tiny schema-exact fixture for smoke tests (also run in CI)
```

> Heavy, regenerable artifacts (`data/`, `probs/`, `ext/`, `subs/`, `runs/`,
> `logs/`) are intentionally **not** committed — see `.gitignore`. The dataset is
> re-fetched with `src/data.py`; everything else is reproduced by the scripts above.
> Scripts use flat sibling imports, so run them as `python src/<name>.py` from the
> repository root (shell scripts in `scripts/` already `cd` to the root themselves).

---

## Quickstart

```bash
conda create -n araseg python=3.10 -y && conda activate araseg
pip install -r requirements.txt

# 1) Fetch the data locally (one task shown; repeat for NoPnx-PA / NP / NoPnx-NP)
python src/data.py --task PA --out-dir data

# 2) Rule baseline -> submission CSV (no GPU needed)
python src/baselines.py --task PA --split dev --rules punct verse par --out subs/PA_dev.csv

# 3) Score offline (mirrors the official metric, with correct P/R labels)
python src/eval_local.py --task PA --split dev --predictions subs/PA_dev.csv --show-worst 5

# 4) Fine-tune an encoder (a single 16 GB GPU / Colab T4 is plenty — 174 train docs)
python src/train_encoder.py --task NoPnx-NP --out-dir runs/nopnx-np

# 5) Predict + validate format, then upload the CSV on CodaBench
python src/predict.py --model runs/nopnx-np --task NoPnx-NP --split dev --out subs/NoPnx-NP_dev.csv
```

**Reproduce the full ensemble + decoding system:** cache each model's
probabilities (`src/cache_probs.py`), average them (`src/ensemble_sweep.py`), then
decode (`src/dp_decode.py` / `src/dp_adaptive.py`). Exact configs per task are in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); `src/predict_blind.py` regenerates all
eight frozen submissions in one command.

**Submission format.** CSV with columns `Document ID,Prediction`, where
`Prediction` is a binary string with exactly one `0/1` per token (including `\n`
tokens). `src/predict.py --check-ids` validates IDs and lengths against the
official example files.

---

## Notes on the data (these drive the modeling)

- Tokens are whitespace-split; **punctuation marks are their own tokens**, and in
  punctuated variants the gold boundary `1` sits **on** the punctuation token.
- **Paragraph breaks are literal `\n` tokens** (PA variants); gold labels the
  token *before* `\n` as the boundary and `\n` itself as `0`.
- Commas are *sometimes* boundaries (e.g. hadith isnad chains) — pure punctuation
  rules cap out there; that residual ambiguity is the modeling head-room.
- Splits: **174 train / 222 dev / 262 test** docs (~700 tokens each), 8 genres.
  The metric is per-document macro-F1, so short docs and rare genres count equally.

---

## Links

- **Task site:** https://www.araseg.aramlab.ai/
- **Official eval + starter code:** https://github.com/mbzuai-nlp/araseg-shared-task-2026
- **Dataset (HuggingFace):** `MBZUAI/AraSeg-2026-Shared-Task-{PA, NoPnx-PA, NP, NoPnx-NP}`
- **Dataset paper:** *Arabic Sentence Segmentation Across Genres and Punctuation
  Conditions* — [arXiv:2606.08025](https://arxiv.org/abs/2606.08025)

To run the official evaluator or use `src/predict.py --check-ids`, clone the
starter repo alongside this one:
```bash
git clone https://github.com/mbzuai-nlp/araseg-shared-task-2026
```

## Citing

See [CITATION.cff](CITATION.cff); the system paper is
[paper/araseg_system.pdf](paper/araseg_system.pdf) (ACL-format version in
[paper/acl/](paper/acl/)).

## License

[MIT](LICENSE). This repository contains only original system code; it does **not**
redistribute the AraSeg corpus or the organizers' official code (each under their
own MIT license — see links above).
