#!/bin/bash
# Random control arms (uniform + depth-matched, seed 0) for ONE model on the
# extraversion task. Called by persona_dispatch.sh, which handles concurrency.
#
# Usage: persona_random_one.sh <model_id> <model_name>
#
# Runs the model's 8 invocations serially within itself: 2 localization trees x
# 2 algos x 2 eval modes, steering matched to eval mode. Already-complete
# invocations are skipped by run.py in ~3 min, so re-running is cheap and safe.
#
# Depth-matched random needs the ATP selection from the SAME localization tree
# (eval/logits_handler.py:112). For persona that lives in the umang checkout and
# is symlinked into results/<model>/from_extraversion-*/atp here.
set -u
model_id="$1"
model_name="$2"

export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
cd /home/ubuntu/gcm-interp

algos=(${ALGOS:-random-s0 randomlayer-s0})
STEERING_FACTORS="${STEERING_FACTORS:-1,2,4,5,6,8,10}"
TOPK_VALS="${TOPK_VALS:-0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0}"
device="cuda:0"

for loc in long single; do
  source="extraversion-${loc}"
  base="introversion-${loc}"
  for algo in "${algos[@]}"; do
    for ev in long single; do
      ed="$RM_INTERP_REPO/data/${model_name}/extraversion-${ev}"
      echo "[$(date '+%F %H:%M:%S')] $model_name  loc=$source  algo=$algo  eval=$ev"
      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source "$source" \
                    --base "$base" \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test "$ed/introversion-${ev}-test.jsonl" \
                    --steering \
                    --ablation steer \
                    --topk_vals "$TOPK_VALS" \
                    --steering_factors "$STEERING_FACTORS" \
                    --steering_add_path "$ed/extraversion-${ev}-steering.jsonl" \
                    --steering_sub_path "$ed/introversion-${ev}-steering.jsonl"
    done
  done
done
echo "[$(date '+%F %H:%M:%S')] MODEL_DONE $model_name"
