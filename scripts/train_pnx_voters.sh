#!/bin/bash
# Train the 5 deployed voter recipes for the Pnx variants (repo tasks NP and PA).
# Append-only out-dirs: runs/pnxnp_<voter>, runs/pnxpa_<voter>. ckpts deleted after each.
cd "C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation" || exit 1
export PYTHONUTF8=1 TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0

train () {   # $1=TASK  $2=voter  $3...=extra flags
  T=$1; V=$2; shift 2
  low=$(echo "$T" | tr '[:upper:]' '[:lower:]')
  OUT="runs/pnx${low}_${V}"
  if [ -f "$OUT/model.safetensors" ]; then echo "SKIP $OUT (exists)"; return; fi
  echo "=== TRAIN $OUT ==="
  python -u src/train_encoder.py --task "$T" \
    --train-jsonl "data/${T}_train.jsonl" --dev-jsonl "data/${T}_dev.jsonl" \
    --epochs 10 --out-dir "$OUT" --save-total-limit 1 "$@" \
    && rm -rf "$OUT/ckpts" && echo "OK $OUT" || echo "FAIL $OUT"
}

for T in NP PA; do
  train "$T" imp_fgm            --model-name aubmindlab/bert-large-arabertv02 --fgm-eps 1.0
  train "$T" nat_arabertv02_base --model-name aubmindlab/bert-base-arabertv02
  train "$T" xlmr-2e5           --model-name FacebookAI/xlm-roberta-large --lr 2e-5 --batch-size 4
  train "$T" boost_supcon01     --model-name aubmindlab/bert-large-arabertv02 --supcon-weight 0.1
  train "$T" nat_asafaya_large  --model-name asafaya/bert-large-arabic
done
echo "ALL_PNX_VOTERS_DONE"
