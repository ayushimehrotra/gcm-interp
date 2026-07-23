export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
echo "RM_INTERP_REPO is $RM_INTERP_REPO"

declare -a pairs=(
  "sycophancy-single_non-sycophantic"
)
declare -A eval_datasets

algos=("atp")
model_id="Qwen/Qwen1.5-14B-Chat"
model_name="Qwen1.5-14B-Chat"
device="cuda:0"

for pair in "${pairs[@]}"; do
    IFS='_' read -r source base <<< "$pair"

  for algo in "${algos[@]}"; do
      python run.py --model_id "$model_id" \
                    --batch_size 4 \
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
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-single/non-sycophantic-test.jsonl"\
                    --steering \
                    --kv_caching \
                    --ablation steer \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-single/non-sycophantic-desired-all.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-single/sycophancy-single-desired-all.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 4 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-single/non-sycophantic-test.jsonl"\
                    --steering \
                    --kv_caching \
                    --ablation steer \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-long/non-sycophantic-desired-all.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-long/sycophancy-long-desired-all.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 4 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-long/non-sycophantic-test.jsonl"\
                    --steering \
                    --kv_caching \
                    --ablation steer \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-single/non-sycophantic-desired-all.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-single/sycophancy-single-desired-all.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 4 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-long/non-sycophantic-test.jsonl"\
                    --steering \
                    --kv_caching \
                    --ablation steer \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-long/non-sycophantic-desired-all.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/sycophancy-long/sycophancy-long-desired-all.jsonl"
  done
done
