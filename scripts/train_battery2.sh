#!/usr/bin/env bash
# Battery 2: corruption-aug models, AraBERT-large, mmBERT, (XLM-R if cached).
set -u
PY="${PY:-python}"
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$(dirname "$0")/.."
mkdir -p runs logs

run() {  # run <tag> <task> <train_jsonl> <extra args...>
  local tag="$1" t="$2" tj="$3"; shift 3
  local out="runs/$(echo "$t" | tr 'A-Z' 'a-z')-${tag}"
  [ -f "$out/config.json" ] && { echo "SKIP $t $tag"; return; }
  echo "===== training $t $tag ====="
  if $PY src/train_encoder.py --task "$t" --train-jsonl "$tj" \
      --dev-jsonl "data/${t}_dev.jsonl" --out-dir "$out" "$@" \
      > "logs/train_${t}_${tag}.log" 2>&1; then
    echo "OK $t $tag"
  else
    echo "FAILED $t $tag (see logs/train_${t}_${tag}.log)"
  fi
}

# 1) corruption-augmented (4x data -> fewer epochs)
run aug NoPnx-PA data/NoPnx-PA_train_aug.jsonl \
    --model-name aubmindlab/bert-base-arabertv02 --epochs 5
run aug NoPnx-NP data/NoPnx-NP_train_aug.jsonl \
    --model-name aubmindlab/bert-base-arabertv02 --epochs 5

# 2) AraBERT-large
for t in PA NoPnx-PA NP NoPnx-NP; do
  run large "$t" "data/${t}_train.jsonl" \
      --model-name aubmindlab/bert-large-arabertv02 --batch-size 8 --lr 2e-5
done

# 3) mmBERT-base (8K context, whole-doc windows)
if [ "${WITH_MMBERT:-1}" = "1" ]; then
  for t in PA NoPnx-PA NP NoPnx-NP; do
    run mmbert "$t" "data/${t}_train.jsonl" \
        --model-name jhu-clsp/mmBERT-base --window 700 --stride 350 \
        --max-length 2048 --batch-size 4 --lr 3e-5
  done
fi

# 4) XLM-R-large (only if fully cached)
if [ "${WITH_XLMR:-0}" = "1" ]; then
  for t in PA NoPnx-PA NP NoPnx-NP; do
    run xlmrl "$t" "data/${t}_train.jsonl" \
        --model-name FacebookAI/xlm-roberta-large --batch-size 4 --lr 2e-5
  done
fi
echo "BATTERY2 DONE"
