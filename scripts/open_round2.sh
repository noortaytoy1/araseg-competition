#!/usr/bin/env bash
# Open-track round 2: attack the NoPnx-NP open ceiling with 4 new voter types
# (classical Tashkeela, 1M-sent scale-up, XLM-R-large, fine-tuned SaT-12L-sm)
# + a classical check on NoPnx-PA. Sequential on one GPU; ckpts cleaned per run.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
V27="${V27:-python}"  # env with torch>=2.6 + wtpsplit + peft
mkdir -p logs open_runs probs

step() { echo "=== [$(date +%H:%M)] $1 ==="; }

step "0. data build (Tashkeela + scale mix)"
$PY src/build_open2_data.py > logs/open2_data.log 2>&1 && echo "OK data" || { echo "FAIL data"; tail -5 logs/open2_data.log; exit 1; }

step "1. classical voter NoPnx-NP (pretrain 3ep -> ft 8ep)"
$PY src/train_encoder.py --task NoPnx-NP --model-name aubmindlab/bert-base-arabertv02 \
    --train-jsonl ext/pretrain_clas_NoPnx-NP.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl \
    --epochs 3 --out-dir open_runs/nopnx-np-claspre > logs/open2_claspre.log 2>&1 \
  && rm -rf open_runs/nopnx-np-claspre/ckpts && echo "OK claspre" || echo "FAIL claspre"
$PY src/train_encoder.py --task NoPnx-NP --model-name open_runs/nopnx-np-claspre \
    --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl \
    --epochs 8 --out-dir open_runs/nopnx-np-clasft > logs/open2_clasft.log 2>&1 \
  && rm -rf open_runs/nopnx-np-clasft/ckpts && echo "OK clasft" || echo "FAIL clasft"

step "2. scale voter NoPnx-NP (40k docs pretrain 2ep -> ft 8ep)"
$PY src/train_encoder.py --task NoPnx-NP --model-name aubmindlab/bert-base-arabertv02 \
    --train-jsonl ext/pretrain_scale_NoPnx-NP.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl \
    --epochs 2 --out-dir open_runs/nopnx-np-scalepre > logs/open2_scalepre.log 2>&1 \
  && rm -rf open_runs/nopnx-np-scalepre/ckpts && echo "OK scalepre" || echo "FAIL scalepre"
$PY src/train_encoder.py --task NoPnx-NP --model-name open_runs/nopnx-np-scalepre \
    --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl \
    --epochs 8 --out-dir open_runs/nopnx-np-scaleft > logs/open2_scaleft.log 2>&1 \
  && rm -rf open_runs/nopnx-np-scaleft/ckpts && echo "OK scaleft" || echo "FAIL scaleft"

step "3. XLM-R-large NoPnx-NP (batch 4)"
if timeout 1800 env HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli \
     download FacebookAI/xlm-roberta-large --quiet > logs/open2_xlmr_dl.log 2>&1; then
  $PY src/train_encoder.py --task NoPnx-NP --model-name FacebookAI/xlm-roberta-large \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl \
      --epochs 8 --batch-size 4 --out-dir open_runs/nopnx-np-xlmrl > logs/open2_xlmrl.log 2>&1 \
    && rm -rf open_runs/nopnx-np-xlmrl/ckpts && echo "OK xlmrl" || echo "FAIL xlmrl (train)"
else
  echo "SKIP xlmrl (download failed again)"
fi

step "4. classical voter NoPnx-PA"
$PY src/train_encoder.py --task NoPnx-PA --model-name aubmindlab/bert-base-arabertv02 \
    --train-jsonl ext/pretrain_clas_NoPnx-PA.jsonl --dev-jsonl data/NoPnx-PA_dev.jsonl \
    --epochs 3 --out-dir open_runs/nopnx-pa-claspre > logs/open2_claspre_pa.log 2>&1 \
  && rm -rf open_runs/nopnx-pa-claspre/ckpts && echo "OK claspre-pa" || echo "FAIL claspre-pa"
$PY src/train_encoder.py --task NoPnx-PA --model-name open_runs/nopnx-pa-claspre \
    --train-jsonl data/NoPnx-PA_train.jsonl --dev-jsonl data/NoPnx-PA_dev.jsonl \
    --epochs 8 --out-dir open_runs/nopnx-pa-clasft > logs/open2_clasft_pa.log 2>&1 \
  && rm -rf open_runs/nopnx-pa-clasft/ckpts && echo "OK clasft-pa" || echo "FAIL clasft-pa"

step "5. SaT-12L-sm fine-tune (venv27)"
$V27 src/train_sat_ft.py > logs/open2_satft.log 2>&1 && echo "OK satft" || { echo "FAIL satft"; tail -5 logs/open2_satft.log; }

step "6. probability caches (new voters dev+test; closed voters test)"
for split in dev test; do
  for v in clasft scaleft xlmrl; do
    [ -f "open_runs/nopnx-np-$v/model.safetensors" ] || continue
    [ -f "probs/NoPnx-NP_${split}_${v}.npz" ] && continue
    $PY src/cache_probs.py --task NoPnx-NP --split $split --models open_runs/nopnx-np-$v \
        --out probs/NoPnx-NP_${split}_${v}.npz >> logs/open2_cache.log 2>&1 && echo "cached $v $split"
  done
done
declare -A CV=( [voter]=runs/nopnx-np [voter-s1]=runs/nopnx-np-s1 [voter-s2]=runs/nopnx-np-s2 \
                [voter-s3]=runs/nopnx-np-s3 [voter-arbertv2]=runs/nopnx-np-arbertv2 \
                [voter-araelectra]=runs/nopnx-np-araelectra )
for name in "${!CV[@]}"; do
  [ -f "probs/NoPnx-NP_test_${name}.npz" ] && continue
  $PY src/cache_probs.py --task NoPnx-NP --split test --models "${CV[$name]}" \
      --out probs/NoPnx-NP_test_${name}.npz >> logs/open2_cache.log 2>&1 && echo "cached $name test"
done
for split in dev test; do
  [ -f "open_runs/nopnx-pa-clasft/model.safetensors" ] || continue
  [ -f "probs/NoPnx-PA_${split}_clasft.npz" ] && continue
  $PY src/cache_probs.py --task NoPnx-PA --split $split --models open_runs/nopnx-pa-clasft \
      --out probs/NoPnx-PA_${split}_clasft.npz >> logs/open2_cache.log 2>&1 && echo "cached pa-clasft $split"
done
for split in dev test; do
  [ -f "probs/NoPnx-PA_${split}_enc3.npz" ] && continue
  $PY src/cache_probs.py --task NoPnx-PA --split $split \
      --models runs/nopnx-pa runs/nopnx-pa-arbertv2 runs/nopnx-pa-araelectra \
      --out probs/NoPnx-PA_${split}_enc3.npz >> logs/open2_cache.log 2>&1 && echo "cached pa-enc3 $split"
done

step "7. selection (greedy dev, test confirm)"
$PY src/open_eval2.py 2>&1 | tee logs/open2_eval.log | tail -30

step "8. NoPnx-PA classical check (3-encoder + clasft @ thr .5 vs frozen)"
$PY - <<'EOF' 2>&1 | tail -6
import numpy as np
from data import load_jsonl
from eval_local import compute_metrics
for split, ref in [("dev", 87.16), ("test", 87.19)]:
    docs = load_jsonl(f"data/NoPnx-PA_{split}.jsonl")
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    import os
    base = [f"probs/NoPnx-PA_{split}_enc3.npz"]
    try:
        pools = {}
        enc = np.load(f"probs/NoPnx-PA_{split}_enc3.npz")
        cla = np.load(f"probs/NoPnx-PA_{split}_clasft.npz")
        for name, mix in [("enc3", lambda d: enc[d]), ("enc3+clasft", lambda d: (3*enc[d]+cla[d])/4)]:
            preds = {d["doc_id"]: [1 if p >= 0.5 else 0 for p in mix(d["doc_id"])] for d in docs}
            print(f"NoPnx-PA {split} {name}: {compute_metrics(gold, preds)['macro_f1']*100:.2f} (frozen {ref})")
    except FileNotFoundError as e:
        print("PA check skipped:", e)
EOF
echo "OPEN ROUND 2 COMPLETE"
