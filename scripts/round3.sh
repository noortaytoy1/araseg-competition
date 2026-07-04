#!/usr/bin/env bash
# Round 3: exploit the proven architecture-diversity lever.
#  1. satft2: longer SaT fine-tune (14 ep; v1 was still climbing at 6)
#  2. satftpa: SaT voter for NoPnx-PA (first arch-diverse candidate there)
#  3. mdeberta / xlmr-base voters for NoPnx-NP + NoPnx-PA (closed-legal!)
#  4. cache probs, then src/round3_eval.py applies the pre-registered rule.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
V27="${V27:-python}"  # env with torch>=2.6 + wtpsplit + peft
mkdir -p logs probs open_runs

step() { echo "=== [$(date +%H:%M)] $1 ==="; }

step "1. satft2 (NoPnx-NP, 14 epochs)"
$V27 src/train_sat_ft.py --task NoPnx-NP --epochs 14 --suffix satft2 \
  > logs/r3_satft2.log 2>&1 && echo "OK satft2" || { echo "FAIL satft2"; tail -3 logs/r3_satft2.log; }

step "2. satftpa (NoPnx-PA, 14 epochs)"
$V27 src/train_sat_ft.py --task NoPnx-PA --epochs 14 --suffix satftpa \
  > logs/r3_satftpa.log 2>&1 && echo "OK satftpa" || { echo "FAIL satftpa"; tail -3 logs/r3_satftpa.log; }

step "3. mdeberta + xlmr-base (NoPnx-NP, NoPnx-PA)"
for enc in microsoft/mdeberta-v3-base FacebookAI/xlm-roberta-base; do
  short=$( [ "$enc" = "microsoft/mdeberta-v3-base" ] && echo mdeberta || echo xlmrb )
  for task in NoPnx-NP NoPnx-PA; do
    low=$(echo $task | tr 'A-Z' 'a-z'); sfx=$( [ "$task" = "NoPnx-PA" ] && echo "-pa" || echo "" )
    out="open_runs/${low}-${short}"
    [ -f "$out/model.safetensors" ] && { echo "SKIP $short $task"; continue; }
    $PY src/train_encoder.py --task "$task" --model-name "$enc" \
        --train-jsonl "data/${task}_train.jsonl" --dev-jsonl "data/${task}_dev.jsonl" \
        --epochs 8 --out-dir "$out" > "logs/r3_${short}_${task}.log" 2>&1 \
      && { rm -rf "$out/ckpts"; echo "OK $short $task"; } \
      || { echo "FAIL $short $task"; tail -3 "logs/r3_${short}_${task}.log"; }
  done
done

step "4. cache probs (dev+test)"
for task in NoPnx-NP NoPnx-PA; do
  low=$(echo $task | tr 'A-Z' 'a-z'); sfx=$( [ "$task" = "NoPnx-PA" ] && echo "-pa" || echo "" )
  for short in mdeberta xlmrb; do
    [ -f "open_runs/${low}-${short}/model.safetensors" ] || continue
    for split in dev test; do
      [ -f "probs/${task}_${split}_${short}${sfx}.npz" ] && continue
      $PY src/cache_probs.py --task "$task" --split "$split" --models "open_runs/${low}-${short}" \
          --out "probs/${task}_${split}_${short}${sfx}.npz" >> logs/r3_cache.log 2>&1 \
        && echo "cached ${short}${sfx} $split"
    done
  done
done

step "5. selection (pre-registered rule)"
$PY src/round3_eval.py 2>&1 | tee logs/r3_eval.log | grep -E "^\[|dev |ADOPT|reject|base"
echo "ROUND 3 COMPLETE"
