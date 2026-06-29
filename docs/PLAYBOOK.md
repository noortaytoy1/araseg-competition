# AraSeg 2026 Playbook — From PC Setup to Submission to Paper

Updated June 12, 2026. Incorporates the organizers' Slack clarifications.
Companion to the starter kit (`README.md`, `data.py`, `baselines.py`,
`train_encoder.py`, `predict.py`, `eval_local.py`).

---

## 0. Ground rules (organizer-confirmed on Slack — these govern everything)

| Allowed | Not allowed |
|---|---|
| Fine-tuning pretrained encoders (CAMeLBERT, AraBERT, MARBERT, AraELECTRA, XLM-R…) in the **closed** track | Fine-tuning on anything other than AraSeg **train** in the closed track |
| Using **dev** for model selection: picking checkpoints, encoders, thresholds | **Training on dev** in any form — both tracks |
| Open track: training on any **external public** data | Using **test inputs** for anything but predictions (no MLM adaptation, no calibration) — both tracks |

Your registration is binding: you must (1) submit in the testing phase
(Jul 20–25) and (2) submit a system description paper (Aug 8), or risk a ban
from future shared tasks. Plan for both from day one.

---

## 1. Today, part 1 — accounts and access (~30 min, no code)

1. **Decide who submits.** Pick ONE CodaBench account as the team's submitting
   account (one teammate is already registered for PA-closed — consider using
   that account for everything to keep one leaderboard identity). Agree on a
   team name and use it consistently.
2. **Register that account in every competition you'll enter** — at minimum
   all four closed-track ones:

   | Subtask | Closed | Open |
   |---|---|---|
   | PA | codabench.org/competitions/16606 | /16607 |
   | NoPnx-PA | codabench.org/competitions/16608 | /16609 |
   | NP | codabench.org/competitions/16610 | /16611 |
   | NoPnx-NP | codabench.org/competitions/16612 | /16613 |

   On each page, request participation and wait for approval (the organizers
   approve manually and actively monitor this). Check each competition's
   page for the **daily submission limit** and budget accordingly.
3. **Slack:** turn on notifications for the workspace — the blind-test release
   and any rule updates will land there and by email.
4. **Calendar blocks now:** Jul 20 (blind test out + registration closes),
   Jul 25 (test submissions close), Aug 8 (paper), Aug 15 (notification),
   Aug 22 (camera-ready = final post-review version of the paper).

---

## 2. Today, part 2 — PC setup (~30–45 min)

Works the same on Windows (PowerShell), Linux, and macOS unless noted.

1. **Install Git and Miniconda** (git-scm.com; docs.conda.io → Miniconda).
   On Windows, do the rest inside "Anaconda PowerShell Prompt".
2. **Environment:**
   ```bash
   conda create -n araseg python=3.10 -y
   conda activate araseg
   ```
3. **Code:**
   ```bash
   git clone https://github.com/mbzuai-nlp/araseg-shared-task-2026
   mkdir araseg-system && cd araseg-system
   # put the starter-kit files here: data.py baselines.py train_encoder.py
   # predict.py eval_local.py requirements.txt
   pip install -r requirements.txt
   ```
4. **GPU (only if you have a local NVIDIA card):** install the CUDA build of
   torch using the selector at pytorch.org → Get Started, then verify:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   No local GPU is fine — training runs on free Colab/Kaggle (step 4 below),
   and everything else in this playbook is CPU-friendly.
5. **Download the data once** (then everything runs offline):
   ```bash
   python src/data.py --task PA --out-dir data
   python src/data.py --task NoPnx-PA --out-dir data
   python src/data.py --task NP --out-dir data
   python src/data.py --task NoPnx-NP --out-dir data
   ```

---

## 3. Today, part 3 — first leaderboard entries (CPU, ~15 min)

1. **Generate rule-baseline predictions** (no training data used → legal in
   both tracks):
   ```bash
   python src/baselines.py --task PA       --split dev --rules punct verse par --out subs/PA_dev.csv
   python src/baselines.py --task NP       --split dev --rules punct verse     --out subs/NP_dev.csv
   python src/baselines.py --task NoPnx-PA --split dev --rules par             --out subs/NoPnx-PA_dev.csv
   python src/baselines.py --task NoPnx-NP --split dev --rules every-k         --out subs/NoPnx-NP_dev.csv
   ```
2. **Score locally before uploading:**
   ```bash
   python src/eval_local.py --task PA --split dev --predictions subs/PA_dev.csv --show-worst 5
   ```
   (You can cross-check with the official `scripts/eval.py`; remember its
   Precision/Recall lines are printed swapped — F1 is correct.)
3. **Submit on CodaBench:** competition page → My Submissions → upload the
   CSV (if the uploader expects an archive, zip the single CSV first — follow
   that competition's Get Started instructions). Wait for it to process and
   show scores, then under **Actions click the first button to publish to the
   leaderboard** (organizer-confirmed flow). During the dev phase you may
   only see the baseline and your own published rows — that's normal.
4. Repeat per task. Expect: PA/NP decent (boundaries sit on punctuation
   tokens), NoPnx-PA moderate (paragraph rule only), NoPnx-NP weak (every-k
   floor). Beating these is what the models are for.

Submitting today does two things: proves your whole pipeline end-to-end, and
shows the organizers you're active (their email noted most registrants
haven't started).

---

## 4. This week — train the four closed-track models

**Pick your compute:**
- Local NVIDIA GPU (≥6 GB VRAM): run commands as-is.
- No GPU → **Google Colab** (free T4): Runtime → Change runtime type → T4,
  then in cells:
  ```
  !pip -q install transformers datasets accelerate scikit-learn pandas
  # upload the starter-kit .py files via the Files sidebar
  !python src/train_encoder.py --task PA --out-dir runs/pa
  !zip -rq pa_model.zip runs/pa   # download, or predict right in Colab
  ```
  Kaggle Notebooks (free weekly GPU quota) works identically.
- CPU-only fallback: feasible (corpus is tiny) — add `--batch-size 8` and
  expect a few hours per task.

**Train (one model per task):**
```bash
python src/train_encoder.py --task PA       --out-dir runs/pa
python src/train_encoder.py --task NoPnx-PA --out-dir runs/nopnx-pa
python src/train_encoder.py --task NP       --out-dir runs/np
python src/train_encoder.py --task NoPnx-NP --out-dir runs/nopnx-np
```

**Predict on dev with format validation, then evaluate:**
```bash
python src/predict.py --model runs/pa --task PA --split dev \
    --out subs/PA_dev_model.csv \
    --check-ids ../araseg-shared-task-2026/examples/PA_dev.csv
python src/eval_local.py --task PA --split dev --predictions subs/PA_dev_model.csv
```

**Tune the threshold on dev** (this is model selection — allowed):
```bash
for t in 0.30 0.40 0.50 0.60 0.70; do
  python src/predict.py --model runs/pa --task PA --split dev --threshold $t --out /tmp/t$t.csv
  echo "threshold $t"; python src/eval_local.py --task PA --split dev --predictions /tmp/t$t.csv
done
```
(PowerShell: `foreach ($t in 0.3,0.4,0.5,0.6,0.7) { ... }`.) Record the best
threshold per task — you'll reuse it on the blind test untouched.

Submit each improved CSV to the matching closed-track competition.

---

## 5. Weeks 2–4 — squeeze more F1

Closed track, roughly in order of payoff:
1. **Encoder swaps:** rerun training with `--model-name
   aubmindlab/bert-base-arabertv02`, `UBC-NLP/ARBERTv2`,
   `aubmindlab/araelectra-base-discriminator`, `xlm-roberta-large` (large
   needs `--batch-size 4` + patience). Keep whichever wins on dev per task.
2. **Seed ensembles:** train 3 seeds (`--seed 1/2/3`), predict each, then
   majority-vote the binary strings:
   ```python
   import pandas as pd, sys
   dfs = [pd.read_csv(p, dtype={"Prediction": str}) for p in sys.argv[1:-1]]
   out = dfs[0][["Document ID"]].copy()
   vote = lambda strs: "".join("1" if sum(s[i]=="1" for s in strs) > len(strs)/2 else "0"
                               for i in range(len(strs[0])))
   out["Prediction"] = [vote([d.loc[i,"Prediction"] for d in dfs]) for i in range(len(out))]
   out.to_csv(sys.argv[-1], index=False)   # usage: python vote.py a.csv b.csv c.csv out.csv
   ```
3. **Sweeps:** `--window 120/180/240`, `--lr 2e-5/5e-5`, `--epochs 8/12`.
4. **PA-variant sanity check:** the token before each `\n` is almost always a
   boundary — verify your model's recall on exactly those positions is ~100%.
5. **Error analysis** (feeds the paper): `eval_local.py --show-worst 10`, read
   those documents, note failure genres (AraSeg spans eight genres and
   cross-genre generalization is the known hard part).

Open track (same four subtasks, separate competitions):
1. **SaT / wtpsplit** (`segment-any-text`): the multilingual no-punctuation
   segmentation SOTA; LoRA-adapt it on AraSeg train.
2. **Self-supervised pretraining on external public corpora only** (Arabic
   Wikipedia, news dumps): split on punctuation → strip it → train the
   boundary classifier to recover it → fine-tune on AraSeg train. Directly
   targets NoPnx-PA / NoPnx-NP. (AraSeg dev/test must never enter training —
   organizer-confirmed.)
3. Your closed-track models can be submitted to the open track too as a
   floor while the fancier systems cook.

---

## 6. Jul 13–19 — freeze and dry-run

1. Lock the final model + threshold per task/track, chosen on dev only.
   Write them in your experiments log.
2. Dry-run the full test pipeline once on the open test split so the blind
   test is mechanical: predict → check-ids-style length sanity → CSV.
3. Re-read each competition's Phases tab so you know exactly where the
   testing-phase upload lives.

---

## 7. Jul 20–25 — testing phase (mandatory)

1. Blind test is released Jul 20 to registered participants — watch email and
   Slack for the access announcement.
2. Whatever format it arrives in, convert each document to one JSON line
   `{"doc_id": "...", "tokens": ["...", ...]}` and run:
   ```bash
   python src/predict.py --model runs/pa --task PA --split test \
       --jsonl blind/PA_test.jsonl --threshold <your-frozen-t> --out subs/PA_test.csv
   ```
   (Labels aren't needed for prediction; the loader doesn't require them.)
3. Submit on **Jul 20–21**, not Jul 25 — leave room for upload hiccups and
   organizer help. Confirm each submission processed and is on the leaderboard.
4. Do not touch thresholds or models based on test inputs.

---

## 8. Jul 26 → Aug 8 — the system description paper

1. You've been logging every run (task, track, encoder, seed, window, lr,
   threshold, dev P/R/F1) since week 1 — that table is most of the paper.
2. Skeleton: Introduction → Task & Data (cite the AraSEG paper,
   arXiv:2606.08025) → System (windowing, [PAR] token, weighted loss,
   thresholds) → Experiments → Results (every task/track you entered) →
   Error analysis (per-genre, punct vs no-punct) → Conclusion.
3. Use the ACL-style ArabicNLP template; the task site's "Shared Task Paper
   Submission" section was "coming soon" — if it still is by mid-July, ask on
   Slack for the submission link and page limit.
4. Aug 15: acceptance + final results. Aug 22: camera-ready — the final
   version after applying reviewer edits (as the organizers explained, no
   cameras involved).
5. The $200 Best System Description Paper award is judged on clarity,
   reproducibility, and insight, independent of leaderboard rank — a clean
   negative result + good error analysis is a real shot at it.

---

## 9. One-page cheat sheet

Daily loop: train/tweak → `predict.py --check-ids` → `eval_local.py` →
submit best to CodaBench → log the run.

Dates: **Jul 20** register-by + blind test out · **Jul 25** test closes ·
**Aug 8** paper · **Aug 15** notification · **Aug 22** camera-ready.

Compliance: pretrained encoders OK everywhere · fine-tune closed-track on
train only · dev = selection only · test = predictions only · external data
open-track only.

Help: Slack #general (organizers respond fast) · araseg26.organizers@aramlab.ai.
