export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"

# Random-head control arms for the long-form localization (CLAUDE.md section 4).
# There is no --patch_model step: the random arms compute no attribution, so the
# eval step is the only one that runs.
#
# The steering vector is matched to the eval mode -- paragraph-long steer with
# paragraph-long eval, paragraph-single steer with paragraph-single eval. That axis is
# deliberately fixed in this paper, so unlike the atp scripts this runs 2 eval
# invocations per algo, not all 4 crossings.
#
# Draw seed and arm are both encoded in --patch_algo, so every arm/seed lands in
# its own results tree (Falcon3-10B-Instruct/from_paragraph-long_to_sentence/<algo>/...) and no
# two draws can overwrite each other. Override the set from the shell, e.g.
#   ALGOS="randomlayer-s0" bash scripts/$(basename "$0")
algos=(${ALGOS:-random-s0 random-s1 random-s2 randomlayer-s0 randomlayer-s1 randomlayer-s2})

# Sweeps default to this model's EXISTING ATP sweep so the arms are comparable.
# The k-curve collapses the N axis by max, so an arm swept over fewer steering
# factors would lose on sweep width rather than on head choice. Note Falcon3-10B-Instruct's
# atp runs cover N=1,2,4,5,6,8,10,15,20 -- config.yml only records the last invocation, so
# read the gen/accuracy filenames rather than the config to confirm this.
STEERING_FACTORS="${STEERING_FACTORS:-1,2,4,5,6,8,10,15,20}"
TOPK_VALS="${TOPK_VALS:-0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0}"

declare -a pairs=(
  "paragraph-long_sentence"
)
declare -A eval_datasets

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
                    --eval_model \
                    --kv_caching \
                    --eval_test  "$RM_INTERP_REPO/data/${model_name}/paragraph-long/sentence-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --topk_vals "$TOPK_VALS" \
                    --steering_factors "$STEERING_FACTORS" \
                    --steering_add_path  "$RM_INTERP_REPO/data/${model_name}/paragraph-long/paragraph-long-desired-all.jsonl" \
                    --steering_sub_path "$RM_INTERP_REPO/data/${model_name}/paragraph-long/sentence-desired-all.jsonl"

      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "$RM_INTERP_REPO/data/${model_name}/paragraph-single/sentence-test.jsonl"\
                    --steering \
                    --ablation steer \
                    --topk_vals "$TOPK_VALS" \
                    --steering_factors "$STEERING_FACTORS" \
                    --steering_add_path  "$RM_INTERP_REPO/data/${model_name}/paragraph-single/paragraph-single-steering.jsonl" \
                    --steering_sub_path "$RM_INTERP_REPO/data/${model_name}/paragraph-single/sentence-single-steering.jsonl"
  done
done
