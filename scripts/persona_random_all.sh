#!/bin/bash
# Random control arms (uniform + depth-matched, seed 0) for the extraversion
# (persona) task, run on THIS repo's pipeline.
#
# Why here and not in gcm-interp-umang: that checkout's eval code calls
# retrieve_random_k() without a seed and has no randomlayer implementation at
# all, so it can only produce uniform random. This repo has is_random(), seeded
# arms, and the depth-matched control -- the more informative of the two, since
# layer placement predicts most of the localization effect.
#
# The persona data lives in the umang checkout and is symlinked into
# data/<model>/extraversion-{long,single}/ here, so data_handler resolves it
# normally (data_path is cwd-relative, config.py:18).
#
# Structure follows the other *_random.sh scripts: for EACH localization tree,
# evaluate in BOTH modes with the steering vector matched to the eval mode.
# Both are needed because the free-form grid plots the depth-matched control
# from each localization arm side by side, and both must be scored on the same
# evaluation as the panel.
#
# Sweeps match the existing ATP persona runs so the arms are comparable:
# N in {1,2,4,5,6,8,10}, topk in {0.01..1.0}.
#
# Falcon3-10B is deliberately excluded: it is dropped from every figure, so its
# runs would buy nothing.
export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
cd /home/ubuntu/gcm-interp

algos=(${ALGOS:-random-s0 randomlayer-s0})
STEERING_FACTORS="${STEERING_FACTORS:-1,2,4,5,6,8,10}"
TOPK_VALS="${TOPK_VALS:-0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0}"
device="cuda:0"

# smallest first, so usable results arrive sooner
declare -a models=(
  "google/gemma-3-12b-it|gemma-3-12b-it"
  "allenai/OLMo-2-1124-13B-DPO|OLMo-2-1124-13B-DPO"
  "Qwen/Qwen1.5-14B-Chat|Qwen1.5-14B-Chat"
  "Qwen/Qwen1.5-32B-Chat|Qwen1.5-32B-Chat"
)

for entry in "${models[@]}"; do
  IFS='|' read -r model_id model_name <<< "$entry"
  for loc in long single; do                    # the localization tree
    source="extraversion-${loc}"
    base="introversion-${loc}"
    for algo in "${algos[@]}"; do
      for ev in long single; do                 # eval mode, steering matched
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
done
echo "[$(date '+%F %H:%M:%S')] PERSONA_RANDOM_DONE"
