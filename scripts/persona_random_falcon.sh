#!/bin/bash
# Random control arms (uniform + depth-matched, seed 0) for Falcon3-10B on the
# extraversion (persona) task.
#
# Separate from persona_random_all.sh because that script is already running and
# editing a live bash script is unsafe -- bash reads it incrementally. This one
# queues behind it.
#
# Falcon's persona data and atp localization both live in THIS repo (unlike the
# other four models, whose persona data is symlinked in from the umang
# checkout), so no linking is needed.
#
# Structure and sweeps match persona_random_all.sh exactly: each localization
# tree evaluated in BOTH modes with the steering vector matched to the eval
# mode; N in {1,2,4,5,6,8,10}, topk in {0.01..1.0}.
export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
cd /home/ubuntu/gcm-interp

algos=(${ALGOS:-random-s0 randomlayer-s0})
STEERING_FACTORS="${STEERING_FACTORS:-1,2,4,5,6,8,10}"
TOPK_VALS="${TOPK_VALS:-0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0}"
device="cuda:0"
model_id="tiiuae/Falcon3-10B-Instruct"
model_name="Falcon3-10B-Instruct"

# wait for the four-model persona queue to finish
pat=$(printf 'persona_random_a%s' 'll.sh')
while pgrep -f "$pat" >/dev/null 2>&1; do sleep 120; done
echo "[$(date '+%F %H:%M:%S')] main persona queue clear, starting Falcon"

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
echo "[$(date '+%F %H:%M:%S')] PERSONA_RANDOM_FALCON_DONE"
