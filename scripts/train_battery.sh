#!/usr/bin/env bash
# Comprehensive battery: AraBERTv02 seeds 1-3 (for seed ensembles),
# then XLM-R-large (capacity), then window-240 AraBERTv02 (context ablation).
set -u
PY="${PY:-python}"
cd "$(dirname "$0")/.."
mkdir -p runs logs

run() {  # run <tag> <task> <extra args...>
  local tag="$1" t="$2"; shift 2
  local out="runs/$(echo "$t" | tr 'A-Z' 'a-z')-${tag}"
  [ -f "$out/config.json" ] && { echo "SKIP $t $tag"; return; }
  echo "===== training $t $tag ====="
  if $PY src/train_encoder.py --task "$t" \
      --train-jsonl "data/${t}_train.jsonl" --dev-jsonl "data/${t}_dev.jsonl" \
      --out-dir "$out" "$@" > "logs/train_${t}_${tag}.log" 2>&1; then
    echo "OK $t $tag"
  else
    echo "FAILED $t $tag (see logs/train_${t}_${tag}.log)"
  fi
}

for s in 1 2 3; do
  for t in PA NoPnx-PA NP NoPnx-NP; do
    run "s${s}" "$t" --model-name aubmindlab/bert-base-arabertv02 --seed "$s"
  done
done

# XLM-R-large skipped 2026-06-12: HF CDN download stalls at ~2.24GB repeatedly.
# Re-enable once the weights are fully cached (huggingface-cli download).
if [ "${WITH_XLMR:-0}" = "1" ]; then
  for t in PA NoPnx-PA NP NoPnx-NP; do
    run "xlmrl" "$t" --model-name FacebookAI/xlm-roberta-large --batch-size 4 --lr 2e-5
  done
fi

for t in PA NoPnx-PA NP NoPnx-NP; do
  run "w240" "$t" --model-name aubmindlab/bert-base-arabertv02 --window 240 --stride 120
done

echo "BATTERY DONE"
