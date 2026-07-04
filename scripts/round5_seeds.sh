#!/usr/bin/env bash
# Round 5: slot-preserving seed averaging (pre-registered in EXPERIMENTS.md).
# Trains the missing seeds for every voter slot across all 4 tasks, caches probs.
# No selection: one variant per task, evaluated later by round5_eval.py.
set -u
cd "$(dirname "$0")"
PY=/home/mohab/Downloads/work/venv/bin/python
V27=/home/mohab/Downloads/COmp-20260611T221353Z-3-001/venv27/bin/python
mkdir -p logs probs

train () { # task model out seed [extra-model-name]
  local task=$1 out=$2 seed=$3 name=$4
  [ -f "runs/$out/model.safetensors" ] || [ -f "open_runs/$out/model.safetensors" ] && { echo "SKIP $out"; return; }
  $PY train_encoder.py --task "$task" --model-name "$name" \
      --train-jsonl "data/${task}_train.jsonl" --dev-jsonl "data/${task}_dev.jsonl" \
      --epochs 8 --seed "$seed" --out-dir "runs/$out" > "logs/r5_${out}.log" 2>&1 \
    && { rm -rf "runs/$out/ckpts"; echo "OK $out"; } || echo "FAIL $out"
}

step() { echo "=== [$(date +%H:%M)] $1 ==="; }

step "1. second/extra seeds for every slot (main venv encoders)"
# PA mega: arabert slot 4 seeds -> 6; arbertv2/araelectra 1 -> 2
train PA pa-s4 4 aubmindlab/bert-base-arabertv02
train PA pa-s5 5 aubmindlab/bert-base-arabertv02
train PA pa-arbertv2-s1 1 UBC-NLP/ARBERTv2
train PA pa-araelectra-s1 1 aubmindlab/araelectra-base-discriminator
# closed NoPnx-NP mega: same
train NoPnx-NP nopnx-np-s4 4 aubmindlab/bert-base-arabertv02
train NoPnx-NP nopnx-np-s5 5 aubmindlab/bert-base-arabertv02
train NoPnx-NP nopnx-np-arbertv2-s1 1 UBC-NLP/ARBERTv2
train NoPnx-NP nopnx-np-araelectra-s1 1 aubmindlab/araelectra-base-discriminator
# NP enc3: seed-2 each
train NP np-s1 1 aubmindlab/bert-base-arabertv02
train NP np-arbertv2-s1 1 UBC-NLP/ARBERTv2
train NP np-araelectra-s1 1 aubmindlab/araelectra-base-discriminator
# NoPnx-PA enc3+mdeberta: seed-2 each (mdeberta needs venv27)
train NoPnx-PA nopnx-pa-s1 1 aubmindlab/bert-base-arabertv02
train NoPnx-PA nopnx-pa-arbertv2-s1 1 UBC-NLP/ARBERTv2
train NoPnx-PA nopnx-pa-araelectra-s1 1 aubmindlab/araelectra-base-discriminator

step "2. mdeberta-pa seed-2 (venv27)"
if [ ! -f runs/nopnx-pa-mdeberta-s1/model.safetensors ]; then
  $V27 train_encoder.py --task NoPnx-PA --model-name microsoft/mdeberta-v3-base \
      --train-jsonl data/NoPnx-PA_train.jsonl --dev-jsonl data/NoPnx-PA_dev.jsonl \
      --epochs 8 --seed 1 --out-dir runs/nopnx-pa-mdeberta-s1 > logs/r5_mdeberta_s1.log 2>&1 \
    && { rm -rf runs/nopnx-pa-mdeberta-s1/ckpts; echo "OK mdeberta-s1"; } || echo "FAIL mdeberta-s1"
fi

step "3. open N-NP slots: opus-ft seed-2 + satft seed-2"
if [ ! -f open_runs/nopnx-np-ft-s1/model.safetensors ]; then
  $PY train_encoder.py --task NoPnx-NP --model-name open_runs/nopnx-np-pre \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl \
      --epochs 8 --seed 1 --out-dir open_runs/nopnx-np-ft-s1 > logs/r5_ft_s1.log 2>&1 \
    && { rm -rf open_runs/nopnx-np-ft-s1/ckpts; echo "OK ft-s1"; } || echo "FAIL ft-s1"
fi
$V27 train_sat_ft.py --task NoPnx-NP --epochs 6 --seed 1 --suffix satft-s1 \
  > logs/r5_satft_s1.log 2>&1 && echo "OK satft-s1" || echo "FAIL satft-s1"

step "4. cache probs (dev+test) for all new models"
declare -A CACHE=(
  [pa-s4]="PA runs/pa-s4" [pa-s5]="PA runs/pa-s5"
  [pa-arbertv2-s1]="PA runs/pa-arbertv2-s1" [pa-araelectra-s1]="PA runs/pa-araelectra-s1"
  [nopnx-np-s4]="NoPnx-NP runs/nopnx-np-s4" [nopnx-np-s5]="NoPnx-NP runs/nopnx-np-s5"
  [nopnx-np-arbertv2-s1]="NoPnx-NP runs/nopnx-np-arbertv2-s1"
  [nopnx-np-araelectra-s1]="NoPnx-NP runs/nopnx-np-araelectra-s1"
  [np-s1]="NP runs/np-s1" [np-arbertv2-s1]="NP runs/np-arbertv2-s1"
  [np-araelectra-s1]="NP runs/np-araelectra-s1"
  [nopnx-pa-s1]="NoPnx-PA runs/nopnx-pa-s1"
  [nopnx-pa-arbertv2-s1]="NoPnx-PA runs/nopnx-pa-arbertv2-s1"
  [nopnx-pa-araelectra-s1]="NoPnx-PA runs/nopnx-pa-araelectra-s1"
  [nopnx-pa-mdeberta-s1]="NoPnx-PA runs/nopnx-pa-mdeberta-s1"
  [nopnx-np-ft-s1]="NoPnx-NP open_runs/nopnx-np-ft-s1"
)
for key in "${!CACHE[@]}"; do
  set -- ${CACHE[$key]}; task=$1; mdir=$2
  [ -f "$mdir/model.safetensors" ] || { echo "nocache $key (missing model)"; continue; }
  for split in dev test; do
    [ -f "probs/${task}_${split}_${key}.npz" ] && continue
    $PY cache_probs.py --task "$task" --split "$split" --models "$mdir" \
        --out "probs/${task}_${split}_${key}.npz" >> logs/r5_cache.log 2>&1 && echo "cached $key $split"
  done
done
echo "ROUND 5 TRAINING+CACHE COMPLETE"
