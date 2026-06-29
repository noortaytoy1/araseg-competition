#!/usr/bin/env bash
# Open-track two-stage pipeline for the NoPnx tasks (most headroom):
#   stage 1: pretrain AraBERTv02 on external boundary-recovery (OPUS sentences)
#   stage 2: fine-tune the pretrained model on AraSeg train
# Dev used only for checkpoint selection (legal). External data = open-track.
set -u
PY="${PY:-python}"
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$(dirname "$0")/.."
mkdir -p open_runs ext

for t in NoPnx-NP NoPnx-PA; do
  echo "===== build pretrain data $t ====="
  $PY src/build_pretrain.py --sents ext/opus_ar_sents.txt --variant "$t" \
      --n-docs 8000 --out "ext/pretrain_${t}.jsonl"

  pre="open_runs/$(echo $t | tr 'A-Z' 'a-z')-pre"
  echo "===== stage1 pretrain $t -> $pre ====="
  $PY src/train_encoder.py --task "$t" --model-name aubmindlab/bert-base-arabertv02 \
      --train-jsonl "ext/pretrain_${t}.jsonl" --dev-jsonl "data/${t}_dev.jsonl" \
      --epochs 3 --out-dir "$pre" > "logs/open_pre_${t}.log" 2>&1 \
    && echo "OK pretrain $t" || { echo "FAIL pretrain $t"; tail -5 "logs/open_pre_${t}.log"; continue; }

  ft="open_runs/$(echo $t | tr 'A-Z' 'a-z')-ft"
  echo "===== stage2 finetune $t -> $ft ====="
  $PY src/train_encoder.py --task "$t" --model-name "$pre" \
      --train-jsonl "data/${t}_train.jsonl" --dev-jsonl "data/${t}_dev.jsonl" \
      --epochs 8 --out-dir "$ft" > "logs/open_ft_${t}.log" 2>&1 \
    && echo "OK finetune $t" || { echo "FAIL finetune $t"; tail -5 "logs/open_ft_${t}.log"; }
done
echo "OPEN PIPELINE DONE"
