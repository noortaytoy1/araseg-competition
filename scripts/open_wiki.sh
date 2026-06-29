#!/usr/bin/env bash
# Third external voter: Wikipedia (formal MSA register) boundary-recovery pretrain
# -> finetune on AraSeg, for both NoPnx tasks.
set -u
PY="${PY:-python}"
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$(dirname "$0")/.."
for t in NoPnx-NP NoPnx-PA; do
  low=$(echo $t | tr 'A-Z' 'a-z')
  $PY src/train_encoder.py --task "$t" --model-name aubmindlab/bert-base-arabertv02 \
      --train-jsonl "ext/pretrain_wiki_${t}.jsonl" --dev-jsonl "data/${t}_dev.jsonl" \
      --epochs 3 --out-dir "open_runs/${low}-wikipre" > "logs/open_wikipre_${t}.log" 2>&1 \
    && echo "OK wikipre $t" || { echo "FAIL wikipre $t"; tail -4 "logs/open_wikipre_${t}.log"; continue; }
  $PY src/train_encoder.py --task "$t" --model-name "open_runs/${low}-wikipre" \
      --train-jsonl "data/${t}_train.jsonl" --dev-jsonl "data/${t}_dev.jsonl" \
      --epochs 8 --out-dir "open_runs/${low}-wikift" > "logs/open_wikift_${t}.log" 2>&1 \
    && echo "OK wikift $t" || { echo "FAIL wikift $t"; tail -4 "logs/open_wikift_${t}.log"; }
done
echo "WIKI PIPELINE DONE"
