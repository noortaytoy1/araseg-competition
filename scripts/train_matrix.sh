#!/usr/bin/env bash
# Train a set of encoders on all 4 tasks: runs/<task>-<tag>, logs/train_<task>_<tag>.log
set -u
PY="${PY:-python}"
cd "$(dirname "$0")/.."
mkdir -p runs logs

# model_id|tag  (safetensors-only models; .bin-only ones fail with torch 2.5)
MODELS=(
  "UBC-NLP/ARBERTv2|arbertv2"
  "aubmindlab/araelectra-base-discriminator|araelectra"
)

for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"; tag="${entry##*|}"
  for t in PA NoPnx-PA NP NoPnx-NP; do
    out="runs/$(echo "$t" | tr 'A-Z' 'a-z')-${tag}"
    [ -f "$out/config.json" ] && { echo "SKIP $t $tag (exists)"; continue; }
    echo "===== training $t $tag ====="
    if $PY src/train_encoder.py --task "$t" --model-name "$model" \
        --train-jsonl "data/${t}_train.jsonl" --dev-jsonl "data/${t}_dev.jsonl" \
        --out-dir "$out" > "logs/train_${t}_${tag}.log" 2>&1; then
      echo "OK $t $tag"
    else
      echo "FAILED $t $tag (see logs/train_${t}_${tag}.log)"
    fi
  done
done
echo "MATRIX DONE"
