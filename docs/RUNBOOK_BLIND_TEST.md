# Blind-Test Runbook (Jul 20–25, 2026)

Verified 2026-06-13: `predict_blind.py` reproduces all 8 frozen submissions
byte-for-byte on the open test. On the day, this is mechanical — no config choices.

## 0. Once, before Jul 20 (dry run)
Confirm the env still works and models are intact:
```
cd ~/Downloads/COmp-20260611T221353Z-3-001/COmp
PY=~/Downloads/work/venv/bin/python
$PY predict_blind.py --track closed --all --blind-dir data --out-dir /tmp/dry/closed
```
Should print 4 lines ending in `-> ...`. If yes, you're ready.

## 1. Jul 20 — get the blind test
Download the blind-test inputs (watch Slack/email), then convert + VALIDATE each
task with `prepare_blind.py` (fails loudly on any tokenisation anomaly — do not
hand-convert):
```
$PY prepare_blind.py --task PA        --in raw/PA_blind.jsonl        --out blind/PA_test.jsonl
$PY prepare_blind.py --task NoPnx-PA  --in raw/NoPnx-PA_blind.jsonl  --out blind/NoPnx-PA_test.jsonl
$PY prepare_blind.py --task NP        --in raw/NP_blind.jsonl        --out blind/NP_test.jsonl
$PY prepare_blind.py --task NoPnx-NP  --in raw/NoPnx-NP_blind.jsonl  --out blind/NoPnx-NP_test.jsonl
$PY prepare_blind.py --check-all --blind-dir blind   # cross-variant invariants
```
(`--raw-dir` instead of `--in` for one-doc-per-file text dumps. The check-all
invariants are verified to hold on the public test set; if doc IDs are de-aliased
across variants in the blind set, per-file checks still run and the cross-check
degrades to a warning.)

## 2. Generate all predictions (three commands)
The open NoPnx-NP config includes a 2-state SaT-ft slot; dump BOTH states from
the venv27 env first:
```
V27=~/Downloads/COmp-20260611T221353Z-3-001/venv27/bin/python
$V27 satft_blind.py --jsonl blind/NoPnx-NP_test.jsonl --out probs/blind_satft.npz
$V27 satft_blind.py --jsonl blind/NoPnx-NP_test.jsonl \
     --state open_runs/nopnx-np-satft-s1/state.pt --out probs/blind_satft_s1.npz
$PY predict_blind.py --track closed --all --blind-dir blind --out-dir blind_closed
$PY predict_blind.py --track open   --all --blind-dir blind --out-dir blind_open \
    --satft-npz probs/blind_satft.npz,probs/blind_satft_s1.npz
```
Sanity: each printed `rate` should be ~0.06–0.10 (boundaries per token). A rate
near 0 or >0.2 means the input wasn't tokenised as expected — fix before uploading.

## 3. Package + upload (8 submissions)
For each CSV: the file inside the zip must be named exactly `prediction`.
```
for f in blind_closed/*.csv; do t=$(basename $f .csv); ( cd blind_closed && cp $t.csv prediction && zip -q ${t}.zip prediction && rm prediction ); done
for f in blind_open/*.csv;   do t=$(basename $f .csv); ( cd blind_open   && cp $t.csv prediction && zip -q ${t}.zip prediction && rm prediction ); done
```
Upload each `prediction.zip` to its competition, then **Actions → first button** to
publish to the leaderboard:

| Task | file | Closed | Open |
|------|------|--------|------|
| PA       | PA.zip       | 16606 | 16607 |
| NoPnx-PA | NoPnx-PA.zip | 16608 | 16609 |
| NP       | NP.zip       | 16610 | 16611 |
| NoPnx-NP | NoPnx-NP.zip | 16612 | 16613 |

## 4. Rules
- Submit Jul 20–21 (not the 25th). Confirm each processed + published.
- Do NOT change any model/threshold/λ based on the blind inputs — predictions only.

## Frozen system (re-frozen 2026-07-05, round-5 slot-preserving seed averaging)
- PA (both tracks): 10 models — 6 AraBERTv02 seeds (slot w=4/6) + 2 ARBERTv2 +
  2 AraELECTRA (each slot w=1/6) — DP decode λ=0.2
- NoPnx-PA (both tracks): 8 models — 2 seeds each of {AraBERTv02, ARBERTv2,
  AraELECTRA, mDeBERTa-v3}, uniform, thr 0.50 +or-par
- NP (both tracks): 6 models — 2 seeds each of the 3 encoders, uniform, thr 0.50
- NoPnx-NP closed: ORIGINAL 6-model mega + adaptive prior (round-5 variant
  breached the dev guard — kept frozen)
- NoPnx-NP open: 4 uniform slots {6 AraBERTv02 seeds, 2 AraELECTRA, 2 OPUS-ft,
  2 SaT-ft states}, adaptive decode (needs the two satft npz dumps — step 2)

Expected ≈ open-test F1: PA 94.43 / NoPnx-PA 87.38 / NP 92.88 / NoPnx-NP 84.97(closed)/85.10(open).
NOTE: re-upload SEVEN zips now (dev phase) — all except closed NoPnx-NP:
  subs/upload/PA          → 16606 | subs/upload_open/PA        → 16607  (94.43)
  subs/upload/NoPnx-PA    → 16608 | subs/upload_open/NoPnx-PA  → 16609  (87.38)
  subs/upload/NP          → 16610 | subs/upload_open/NP        → 16611  (92.88)
  subs/upload_open/NoPnx-NP → 16613  (85.10)
