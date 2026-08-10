export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"

# Random-head control arms for the long-form localization (CLAUDE.md section 4).
# There is no --patch_model step: the random arms compute no attribution, so the
# eval step is the only one that runs.
#
# The steering vector is matched to the eval mode -- verse-long steer with
# verse-long eval, verse-single steer with verse-single eval. That axis is
# deliberately fixed in this paper, so unlike the atp scripts this runs 2 eval
# invocations per algo, not all 4 crossings.
#
# Draw seed and arm are both encoded in --patch_algo, so every arm/seed lands in
# its own results tree (gemma-3-12b-it/from_verse-long_to_prose/<algo>/...) and no
# two draws can overwrite each other. Override the set from the shell, e.g.
#   ALGOS="randomlayer-s0" bash scripts/$(basename "$0")
algos=(${ALGOS:-random-s0 random-s1 random-s2 randomlayer-s0 randomlayer-s1 randomlayer-s2})

# Sweeps default to this model's EXISTING ATP sweep so the arms are comparable.
# The k-curve collapses the N axis by max, so an arm swept over fewer steering
# factors would lose on sweep width rather than on head choice. Note gemma-3-12b-it's
# atp runs cover N=1,2,4,5,6,8,10 -- config.yml only records the last invocation, so
# read the gen/accuracy filenames rather than the config to confirm this.
STEERING_FACTORS="${STEERING_FACTORS:-1,2,4,5,6,8,10}"
TOPK_VALS="${TOPK_VALS:-0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0}"

declare -a pairs=(
  "verse-long_prose"
)
declare -A eval_datasets

model_id="google/gemma-3-12b-it"
model_name="gemma-3-12b-it"
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
                    --eval_model \
                    --kv_caching \
                    --eval_test  "$RM_INTERP_REPO/data/${model_name}/verse-long/prose-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --topk_vals "$TOPK_VALS" \
                    --steering_factors "$STEERING_FACTORS" \
                    --steering_add_path  "$RM_INTERP_REPO/data/${model_name}/verse-long/verse-long-desired-all.jsonl" \
                    --steering_sub_path "$RM_INTERP_REPO/data/${model_name}/verse-long/prose-desired-all.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "$RM_INTERP_REPO/data/${model_name}/verse-single/prose-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --topk_vals "$TOPK_VALS" \
                    --steering_factors "$STEERING_FACTORS" \
                    --steering_add_path  "$RM_INTERP_REPO/data/${model_name}/verse-single/verse-single-steering.jsonl" \
                    --steering_sub_path "$RM_INTERP_REPO/data/${model_name}/verse-single/prose-single-steering.jsonl"
  done
done
