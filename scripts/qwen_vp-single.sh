export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib

declare -a pairs=(
  "verse-single_prose"
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
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/verse-single/prose-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/verse-single/verse-single-steering.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/verse-single/prose-single-steering.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/verse-single/prose-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/verse-long/verse-long-desired-all.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/verse-long/prose-desired-all.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/verse-long/prose-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/verse-single/verse-single-steering.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/verse-single/prose-single-steering.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "/home/ubuntu/gcm-interp/data/${model_name}/verse-long/prose-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --steering_add_path  "/home/ubuntu/gcm-interp/data/${model_name}/verse-long/verse-long-desired-all.jsonl" \
                    --steering_sub_path "/home/ubuntu/gcm-interp/data/${model_name}/verse-long/prose-desired-all.jsonl"
  done
done