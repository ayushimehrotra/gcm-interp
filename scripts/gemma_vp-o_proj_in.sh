export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
export PATH="$RM_INTERP_REPO/.venv/bin:$PATH"

# Verse, gemma-3-12b-it, localized and steered on o_proj.INPUT (--patch_site
# o_proj_in): the concatenated per-head attention outputs, where block u is head
# u. The default site, o_proj.output, is W_O @ concat(z), so every coordinate
# there is already a sum over all heads.
#
# gemma-3-12b is the model where the two sites differ in WIDTH, not just in which
# tensor: hidden_size/num_heads = 3840/16 = 240, but the true head_dim is 256, so
# an o_proj.output block is 240 residual coordinates that straddle two attention
# heads and belong to neither. On o_proj.input the axis is 16 x 256 = 4096 and
# block u is exactly head u. Unit COUNT per layer is 16 either way, so the k-grid
# still lines up with the existing atp runs (48 x 16 = 768 units).
#
# Both localizations, one steering/eval mode:
#   verse-long_prose    long-form localization
#   verse-single_prose  single-token localization
#   both evaluated long-form, steered with the long-form vector (matched, per
#   CLAUDE.md section 4). The atp scripts run all four eval x steer crossings;
#   this one deliberately runs only the long/long cell.
#
# Results land in results/gemma-3-12b-it/from_*_to_prose/atp-o_proj_in/, a
# separate tree from atp/, so nothing here can touch the published o_proj.output
# numbers. Resume is filename-based, so re-running skips finished conditions.

declare -a pairs=(
  "verse-long_prose"
  "verse-single_prose"
)

algos=("atp")
model_id="google/gemma-3-12b-it"
model_name="gemma-3-12b-it"
device="cuda:0"
site="o_proj_in"

# Match this model's existing atp sweep exactly, or the arms are not comparable.
STEERING_FACTORS="${STEERING_FACTORS:-1,2,4,5,6,8,10}"
TOPK_VALS="${TOPK_VALS:-0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0}"

for pair in "${pairs[@]}"; do
    IFS='_' read -r source base <<< "$pair"

  for algo in "${algos[@]}"; do
      echo "=== [$(date -Is)] ATTRIBUTION  $source -> $base  site=$site ==="
      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --patch_site "$site" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --patch_model || { echo "ATTRIBUTION FAILED for $pair"; exit 1; }

      echo "=== [$(date -Is)] EVAL long-steer/long-eval  $source -> $base  site=$site ==="
      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --patch_site "$site" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --eval_model \
                    --kv_caching \
                    --eval_test  "$RM_INTERP_REPO/data/${model_name}/verse-long/prose-test.jsonl" \
                    --steering \
                    --ablation steer \
                    --topk_vals "$TOPK_VALS" \
                    --steering_factors "$STEERING_FACTORS" \
                    --steering_add_path  "$RM_INTERP_REPO/data/${model_name}/verse-long/verse-long-desired-all.jsonl" \
                    --steering_sub_path "$RM_INTERP_REPO/data/${model_name}/verse-long/prose-desired-all.jsonl" \
                    || { echo "EVAL FAILED for $pair"; exit 1; }
  done
done
echo "=== [$(date -Is)] DONE ==="
