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
Download the blind-test inputs (watch Slack/email). Convert each task's docs to
one JSON line per document — NO labels needed:
`{"doc_id": "...", "tokens": ["...", ...]}`
Save as: `blind/PA_test.jsonl`, `blind/NoPnx-PA_test.jsonl`,
`blind/NP_test.jsonl`, `blind/NoPnx-NP_test.jsonl`
(If they arrive as a HuggingFace split: `load_dataset(...); [{"doc_id":d["doc_id"],
"tokens":d["tokens"]} ...]` → write jsonl. If CSV/raw: whitespace-tokenise,
keep "\n" as its own token for the PA variants.)

## 2. Generate all predictions (two commands)
```
$PY predict_blind.py --track closed --all --blind-dir blind --out-dir blind_closed
$PY predict_blind.py --track open   --all --blind-dir blind --out-dir blind_open
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

## Frozen system (what each submission is)
- PA: 6-model mega-ensemble + DP decode (λ=0.2)
- NoPnx-PA: 3-encoder ensemble, thr 0.50, +or-par  ← bootstrap-selected (simpler)
- NP: 3-encoder ensemble, thr 0.50                ← bootstrap-selected (simpler)
- NoPnx-NP: mega + adaptive length prior (closed); +2 external voters (open)

Expected ≈ open-test F1: PA 94.4 / NoPnx-PA 87.2 / NP 92.7 / NoPnx-NP 85.0(open)/84.97(closed).
