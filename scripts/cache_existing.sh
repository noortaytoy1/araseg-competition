#!/usr/bin/env bash
# Cache per-voter probs for the existing PA / NP / NoPnx-PA models (round-5 eval
# needs individual slot caches; only aggregates existed for these tasks).
set -u
cd "$(dirname "$0")"
PY=/home/mohab/Downloads/work/venv/bin/python
declare -A M=(
  [pa-voter]="PA runs/pa" [pa-voter-s1]="PA runs/pa-s1" [pa-voter-s2]="PA runs/pa-s2"
  [pa-voter-s3]="PA runs/pa-s3" [pa-voter-arbertv2]="PA runs/pa-arbertv2"
  [pa-voter-araelectra]="PA runs/pa-araelectra"
  [np-voter]="NP runs/np" [np-voter-arbertv2]="NP runs/np-arbertv2"
  [np-voter-araelectra]="NP runs/np-araelectra"
  [npa-voter]="NoPnx-PA runs/nopnx-pa" [npa-voter-arbertv2]="NoPnx-PA runs/nopnx-pa-arbertv2"
  [npa-voter-araelectra]="NoPnx-PA runs/nopnx-pa-araelectra"
)
for key in "${!M[@]}"; do
  set -- ${M[$key]}; task=$1; mdir=$2
  for split in dev test; do
    [ -f "probs/${task}_${split}_${key}.npz" ] && continue
    $PY cache_probs.py --task "$task" --split "$split" --models "$mdir" \
        --out "probs/${task}_${split}_${key}.npz" >> logs/r5_cache.log 2>&1 && echo "cached $key $split"
  done
done
echo "EXISTING-VOTER CACHE COMPLETE"
