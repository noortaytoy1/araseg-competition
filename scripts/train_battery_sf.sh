#!/usr/bin/env bash
# Smooth+FGM full-pool retrain: all 6 voters per task get boundary label
# smoothing (eps=0.1) + FGM (eps=1.0). Tests whether a pool of better-trained
# voters beats the saturated baseline pool. NoPnx tasks first (most headroom).
set -u
PY="${PY:-python}"
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$(dirname "$0")/.."
mkdir -p runs logs
SF="--smooth-eps 0.1 --fgm-eps 1.0"

run() {  # run <tag> <task> <model> <extra...>
  local tag="$1" t="$2" model="$3"; shift 3
  local out="runs/$(echo "$t" | tr 'A-Z' 'a-z')-${tag}"
  [ -f "$out/config.json" ] && { echo "SKIP $t $tag"; return; }
  echo "===== training $t $tag ====="
  if $PY src/train_encoder.py --task "$t" --model-name "$model" $SF "$@" \
      --train-jsonl "data/${t}_train.jsonl" --dev-jsonl "data/${t}_dev.jsonl" \
      --out-dir "$out" > "logs/train_${t}_${tag}.log" 2>&1; then
    echo "OK $t $tag"
  else
    echo "FAILED $t $tag (see logs/train_${t}_${tag}.log)"
  fi
}

AB=aubmindlab/bert-base-arabertv02
for t in NoPnx-NP NoPnx-PA NP PA; do
  run sf       "$t" "$AB" --seed 42       # seed-42 (exists for NoPnx-*)
  run sf-s1    "$t" "$AB" --seed 1
  run sf-s2    "$t" "$AB" --seed 2
  run sf-s3    "$t" "$AB" --seed 3
  run arbertv2-sf  "$t" UBC-NLP/ARBERTv2
  run araelectra-sf "$t" aubmindlab/araelectra-base-discriminator
done
echo "SF BATTERY DONE"
