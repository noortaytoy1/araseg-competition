#!/usr/bin/env bash
# Data-scaling curve: train single AraBERTv02 on n train docs, eval on dev.
set -u
PY="${PY:-python}"
cd "$(dirname "$0")/.."; mkdir -p runs logs scaling
for task in PA NoPnx-NP; do
  low=$(echo $task | tr 'A-Z' 'a-z')
  for n in 22 44 87 130 174; do
    out="runs/${low}-n${n}"
    [ -f "$out/config.json" ] && { echo "SKIP $task n$n"; continue; }
    $PY src/train_encoder.py --task "$task" --model-name aubmindlab/bert-base-arabertv02 \
        --train-jsonl "data/${task}_train_n${n}.jsonl" --dev-jsonl "data/${task}_dev.jsonl" \
        --epochs 8 --out-dir "$out" > "logs/scale_${task}_n${n}.log" 2>&1 \
      && echo "OK $task n$n" || echo "FAIL $task n$n"
  done
done
echo "SCALING DONE"
