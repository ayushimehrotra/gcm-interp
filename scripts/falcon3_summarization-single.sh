export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib

# Steering factors (N: the multiplier on the normalized steering vector) to sweep.
# Override from the shell, e.g.
#   STEERING_FACTORS="15,20,30,50" bash scripts/$(basename "$0")
# Leave unset to use run.py's default sweep (1,2,4,5,6,8,10).
sf_args=()
if [[ -n "${STEERING_FACTORS:-}" ]]; then
  sf_args=(--steering_factors "$STEERING_FACTORS")
fi

declare -a pairs=(
  "paragraph-single_sentence"
)
declare -A eval_datasets

algos=("atp")
model_id="tiiuae/Falcon3-10B-Instruct"
model_name="Falcon3-10B-Instruct"
device="cuda:0"

for pair in "${pairs[@]}"; do
    IFS='_' read -r source base <<< "$pair"
  
  for algo in "${algos[@]}"; do
      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --patch_model
      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-single/sentence-test.jsonl"\
                    --steering \
                    --ablation steer "${sf_args[@]}" \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-single/paragraph-single-steering.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-single/sentence-single-steering.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-single/sentence-test.jsonl"\
                    --steering \
                    --ablation steer "${sf_args[@]}" \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-long/paragraph-long-desired-all.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-long/sentence-desired-all.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-long/sentence-test.jsonl"\
                    --steering \
                    --ablation steer "${sf_args[@]}" \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-single/paragraph-single-steering.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-single/sentence-single-steering.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-long/sentence-test.jsonl"\
                    --steering \
                    --ablation steer "${sf_args[@]}" \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-long/paragraph-long-desired-all.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/paragraph-long/sentence-desired-all.jsonl"
  done
done