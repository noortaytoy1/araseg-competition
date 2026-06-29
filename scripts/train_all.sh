#!/usr/bin/env bash
# Train all 4 closed-track models sequentially. Detach-safe:
#   nohup bash scripts/train_all.sh > logs/train_all.out 2>&1 &
set -u
PY="${PY:-python}"
# venv has torch 2.5.1: transformers refuses .bin checkpoints (CVE-2025-32434),
# so the encoder must have safetensors weights on the Hub (CAMeLBERT does not).
MODEL=${MODEL:-aubmindlab/bert-base-arabertv02}
cd "$(dirname "$0")/.."
mkdir -p runs logs
rm -f logs/DONE
for t in PA NoPnx-PA NP NoPnx-NP; do
  out=runs/$(echo "$t" | tr 'A-Z' 'a-z')
  echo "===== training $t -> $out ====="
  if $PY src/train_encoder.py --task "$t" --model-name "$MODEL" \
      --train-jsonl "data/${t}_train.jsonl" --dev-jsonl "data/${t}_dev.jsonl" \
      --out-dir "$out" > "logs/train_${t}.log" 2>&1; then
    echo "OK $t"
  else
    echo "FAILED $t (see logs/train_${t}.log)"
  fi
done
touch logs/DONE
echo "ALL DONE"
