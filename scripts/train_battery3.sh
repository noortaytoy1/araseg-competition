#!/usr/bin/env bash
# Battery 3 (venv27, torch>=2.6): CAMeLBERT-MSA + CAMeLBERT-CA on all 4 tasks.
set -u
PY="${PY:-python}"
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$(dirname "$0")/.."
mkdir -p runs logs

run() {
  local tag="$1" t="$2" model="$3"; shift 3
  local out="runs/$(echo "$t" | tr 'A-Z' 'a-z')-${tag}"
  [ -f "$out/config.json" ] && { echo "SKIP $t $tag"; return; }
  echo "===== training $t $tag ====="
  if $PY src/train_encoder.py --task "$t" --model-name "$model" \
      --train-jsonl "data/${t}_train.jsonl" --dev-jsonl "data/${t}_dev.jsonl" \
      --out-dir "$out" "$@" > "logs/train_${t}_${tag}.log" 2>&1; then
    echo "OK $t $tag"
  else
    echo "FAILED $t $tag (see logs/train_${t}_${tag}.log)"
  fi
}

for t in PA NoPnx-PA NP NoPnx-NP; do
  run camel "$t" CAMeL-Lab/bert-base-arabic-camelbert-msa
done
for t in PA NoPnx-PA NP NoPnx-NP; do
  run camelca "$t" CAMeL-Lab/bert-base-arabic-camelbert-ca
done
# mmBERT is .bin-only on the Hub -> needs torch>=2.6, so it lives here
for t in PA NoPnx-PA NP NoPnx-NP; do
  run mmbert "$t" jhu-clsp/mmBERT-base \
      --window 700 --stride 350 --max-length 2048 --batch-size 4 --lr 3e-5
done
echo "BATTERY3 DONE"
