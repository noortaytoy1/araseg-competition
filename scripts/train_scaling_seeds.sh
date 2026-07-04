#!/usr/bin/env bash
# Multi-seed scaling (seeds 1,2; seed 42 already trained) for confidence bands.
# Cleans ckpts after each run to avoid disk pressure.
set -u
PY="${PY:-python}"
cd "$(dirname "$0")/.."; mkdir -p runs logs
for seed in 1 2; do
  for task in PA NoPnx-NP; do
    low=$(echo $task | tr 'A-Z' 'a-z')
    for n in 22 44 87 130 174; do
      out="runs/${low}-n${n}-s${seed}"
      [ -f "$out/model.safetensors" ] && { echo "SKIP $task n$n s$seed"; continue; }
      $PY src/train_encoder.py --task "$task" --model-name aubmindlab/bert-base-arabertv02 \
          --train-jsonl "data/${task}_train_n${n}.jsonl" --dev-jsonl "data/${task}_dev.jsonl" \
          --epochs 8 --seed "$seed" --out-dir "$out" > "logs/scale_${task}_n${n}_s${seed}.log" 2>&1 \
        && { rm -rf "$out/ckpts"; echo "OK $task n$n s$seed"; } || echo "FAIL $task n$n s$seed"
    done
  done
done
echo "SCALING SEEDS DONE"
