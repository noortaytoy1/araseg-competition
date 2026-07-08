#!/usr/bin/env bash
# Hierarchical doc->paragraph->sentence CASCADE on NoPnx-NP (single AraBERTv02).
# Adds NEW structural signal (predicted paragraphs as [PAR] input) the ensemble+DP
# lacks. Self-validating: a GOLD-paragraph ceiling confirms the pipeline before
# the realistic PREDICTED-paragraph result.
set -e
cd "$(dirname "$0")"
T=NoPnx-NP
B=runs/opus_cascade_2026-07-08
mkdir -p "$B"
TR=data/${T}_train.jsonl; DV=data/${T}_dev.jsonl
C="--task $T --epochs 8 --seed 42"
tr_enc () { PYTHONUTF8=1 python src/train_encoder.py $C --train-jsonl "$1" --dev-jsonl "$2" --out-dir "$3" && rm -rf "$3/ckpts"; }

echo "== 0. gold paragraph targets =="
PYTHONUTF8=1 python src/hier_cascade.py para-target $T train data/paratgt_${T}_train.jsonl
PYTHONUTF8=1 python src/hier_cascade.py para-target $T dev   data/paratgt_${T}_dev.jsonl
PYTHONUTF8=1 python src/hier_cascade.py para-csv    $T train "$B/goldpara_train.csv"
PYTHONUTF8=1 python src/hier_cascade.py para-csv    $T dev   "$B/goldpara_dev.csv"

echo "== 1. BASELINE (no paragraphs) =="
tr_enc "$TR" "$DV" "$B/base"
PYTHONUTF8=1 python src/predict.py --model "$B/base" --task $T --split dev --jsonl "$DV" --out "$B/base_dev.csv"
PYTHONUTF8=1 python src/eval_local.py --predictions "$B/base_dev.csv" --task $T --split dev --gold-jsonl "$DV" | tee "$B/base_eval.txt"

echo "== 2. GOLD-PARA CEILING (validates pipeline; should approach NoPnx-PA ~86) =="
PYTHONUTF8=1 python src/hier_cascade.py inject $T train "$B/goldpara_train.csv" data/hiergold_${T}_train.jsonl
PYTHONUTF8=1 python src/hier_cascade.py inject $T dev   "$B/goldpara_dev.csv"   data/hiergold_${T}_dev.jsonl
tr_enc data/hiergold_${T}_train.jsonl data/hiergold_${T}_dev.jsonl "$B/sent_gold"
PYTHONUTF8=1 python src/predict.py --model "$B/sent_gold" --task $T --split dev --jsonl data/hiergold_${T}_dev.jsonl --out "$B/sent_gold_dev.csv"
PYTHONUTF8=1 python src/hier_cascade.py score $T dev data/hiergold_${T}_dev.jsonl "$B/sent_gold_dev.csv" | tee "$B/gold_eval.txt"

echo "== 3. PARAGRAPH PREDICTOR (OUT-OF-FOLD on train so train [PAR] noise == dev) =="
K=3
PYTHONUTF8=1 python src/hier_cascade.py make-folds data/paratgt_${T}_train.jsonl $K "$B/fold"
FOLDCSVS=""
for k in $(seq 0 $((K-1))); do
  # train on other folds, SELECT on real dev (never the held fold), predict held fold
  tr_enc "$B/fold_train_${k}.jsonl" data/paratgt_${T}_dev.jsonl "$B/parafold_${k}"
  PYTHONUTF8=1 python src/predict.py --model "$B/parafold_${k}" --task $T --split train --jsonl "$B/fold_held_${k}.jsonl" --out "$B/parapred_fold_${k}.csv"
  FOLDCSVS="$FOLDCSVS $B/parapred_fold_${k}.csv"
done
PYTHONUTF8=1 python src/hier_cascade.py concat "$B/parapred_train.csv" $FOLDCSVS
# final predictor on ALL train -> predict dev
tr_enc data/paratgt_${T}_train.jsonl data/paratgt_${T}_dev.jsonl "$B/para_pred"
PYTHONUTF8=1 python src/predict.py --model "$B/para_pred" --task $T --split dev --jsonl "$DV" --out "$B/parapred_dev.csv"

echo "== 4. PREDICTED-PARA CASCADE (the real result) =="
PYTHONUTF8=1 python src/hier_cascade.py inject $T train "$B/parapred_train.csv" data/hierpred_${T}_train.jsonl
PYTHONUTF8=1 python src/hier_cascade.py inject $T dev   "$B/parapred_dev.csv"   data/hierpred_${T}_dev.jsonl
tr_enc data/hierpred_${T}_train.jsonl data/hierpred_${T}_dev.jsonl "$B/sent_pred"
PYTHONUTF8=1 python src/predict.py --model "$B/sent_pred" --task $T --split dev --jsonl data/hierpred_${T}_dev.jsonl --out "$B/sent_pred_dev.csv"
PYTHONUTF8=1 python src/hier_cascade.py score $T dev data/hierpred_${T}_dev.jsonl "$B/sent_pred_dev.csv" | tee "$B/pred_eval.txt"

echo ""; echo "===== CASCADE SUMMARY (NoPnx-NP dev macro-F1) ====="
echo -n "  baseline (no para) : "; grep -oE "^F1:[ ]+[0-9.]+" "$B/base_eval.txt" | grep -oE "[0-9.]+$"
echo -n "  gold-para ceiling  : "; grep -oE "[0-9.]+$" "$B/gold_eval.txt" | tail -1
echo -n "  predicted cascade  : "; grep -oE "[0-9.]+$" "$B/pred_eval.txt" | tail -1
