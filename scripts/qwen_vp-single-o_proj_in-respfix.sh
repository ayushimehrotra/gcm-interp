export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
echo "RM_INTERP_REPO is $RM_INTERP_REPO"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
# The venv's bin must be on PATH, not just its python: vLLM shells out to `ninja`
# during CUDA graph capture, so invoking .venv/bin/python directly fails
# (CLAUDE.md section 7). Only the judge needs vLLM, but keep it uniform.
export PATH="$RM_INTERP_REPO/.venv/bin:$PATH"

# Verse, Qwen1.5-14B-Chat, SINGLE-TOKEN localization, on o_proj.INPUT, scoring
# the FULL response span. Three non-default choices, all deliberate:
#
#   source = verse-single     Single-token localization: ATP is computed on the
#       one-letter MCQA turn. Evaluated long-form with the long-form steering
#       vector, which is the same cell scripts/qwen_vp-o_proj_in.sh runs for the
#       shipped metric -- so this run is directly comparable to it, one flag
#       apart. Eval and steer stay matched to each other (long/long), which is
#       the axis CLAUDE.md section 4 fixes; the localization is the thing that
#       differs.
#
#   --patch_site o_proj_in    Score and steer concat(z_1..z_H), the per-head
#       attention outputs, where block u IS head u (head_dim = 5120/40 = 128 for
#       this model, so the unit count per layer -- and therefore the k-grid --
#       matches the existing atp runs exactly). The default site, o_proj.output,
#       is W_O @ concat(z), so every coordinate there is already a sum over ALL
#       heads and a block is residual-stream coordinates that no head owns
#       (FINDINGS 0.1).
#
#   --response_span full      Score the whole assistant response, INCLUDING its
#       first token. The default `legacy` starts one position late. For -long
#       data that drops 1 token of ~127; for -single data the assistant turn is
#       a single letter, so the dropped token is the entire answer and the
#       metric collapses to how readily the model closes the turn after each
#       letter -- the letter is conditioned on but never scored
#       (eval/response_span.py). Correcting that is the point of this run, which
#       is why the localization here is the -single one.
#
# Both flags append to the patch_algo DIRECTORY, so results go to
#   results/Qwen1.5-14B-Chat/from_verse-single_to_prose/atp-o_proj_in-respfix/
# a tree of its own. Nothing here can touch atp/ or atp-o_proj_in/. Resume is
# filename-based, so re-running skips conditions that already finished -- but if
# the metric or sampling logic changes, DELETE the tree rather than resuming, or
# stale results are silently reused.

declare -a pairs=(
  "verse-single_prose"
)

algos=("atp")
# Model is selectable so the same pinned recipe runs on a second model without a
# forked copy of this script. Default is Qwen, so an argument-less invocation is
# byte-identical to what produced the existing tree.
#   MODEL=gemma bash scripts/qwen_vp-single-o_proj_in-respfix.sh
# gemma-3-12b-it is a GATED repo: export HF_TOKEN before running, or the weights
# download 401s. It is also the model where the two sites differ in WIDTH --
# hidden/num_heads = 3840/16 = 240 but head_dim is 256 -- so o_proj_in blocks are
# 256 wide here while o_proj_out blocks are 240. Unit COUNT per layer is 16
# either way (48 x 16 = 768 units), so the k-grid still matches its atp runs.
MODEL="${MODEL:-qwen}"
case "$MODEL" in
  qwen)  model_id="Qwen/Qwen1.5-14B-Chat"; model_name="Qwen1.5-14B-Chat" ;;
  gemma) model_id="google/gemma-3-12b-it"; model_name="gemma-3-12b-it" ;;
  *) echo "Unknown MODEL '$MODEL' (expected: qwen, gemma)"; exit 1 ;;
esac
echo "MODEL=$MODEL -> $model_id"
device="cuda:0"
site="o_proj_in"
span="full"

# Match this model's existing atp sweep exactly, or the arms are not comparable.
STEERING_FACTORS="${STEERING_FACTORS:-1,2,4,5,6,8,10}"
TOPK_VALS="${TOPK_VALS:-0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0}"

for pair in "${pairs[@]}"; do
    IFS='_' read -r source base <<< "$pair"

  for algo in "${algos[@]}"; do
      echo "=== [$(date -Is)] ATTRIBUTION  $source -> $base  site=$site span=$span ==="
      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --patch_site "$site" \
                    --response_span "$span" \
                    --source $source \
                    --base  $base \
                    --device "$device" \
                    --patch_model || { echo "ATTRIBUTION FAILED for $pair"; exit 1; }

      echo "=== [$(date -Is)] EVAL long-steer/long-eval  $source -> $base  site=$site span=$span ==="
      python run.py --model_id "$model_id" \
                    --batch_size 1 \
                    --patch_algo "$algo" \
                    --patch_site "$site" \
                    --response_span "$span" \
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

# Scoring. A long-form eval is judged (judge + fluency + relevance -> w_rf), not
# token-matched (CLAUDE.md section 5). Off by default because it pulls a second
# model -- unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit, ~40 GB -- and takes
# hours. Turn it on with RUN_JUDGE=1.
#
# --no_judge_prefill and --batch_size 16 are pinned to the atp runs and are not
# free choices: the default '(' prefill shifts ratings down a step and w_rf
# counts only rating == 5, so omitting the flag yields plausible near-zero
# accuracies rather than an error. run_judge.py also exits 0 when it prepares
# nothing, so check the "Phase 1 done: N files prepared" line, never rc alone.
# METHOD keeps the full directory name, so accuracies land under
#   judge-evals/accuracy/Qwen1.5-14B-Chat/from_verse-single_to_prose/atp-o_proj_in-respfix/
# and can never merge into the atp tree.
RUN_JUDGE="${RUN_JUDGE:-0}"
if [ "$RUN_JUDGE" = "1" ]; then
  echo "=== [$(date -Is)] JUDGE long-form eval ==="
  export VLLM_USE_DEEP_GEMM=0
  ( cd "$RM_INTERP_REPO/judge-evals" && python run_judge.py \
      --model_name "$model_name" \
      --source verse-single \
      --base prose \
      --eval_subdir verse-long_eval \
      --algos atp-o_proj_in-respfix \
      --batch_size 16 \
      --no_judge_prefill ) || { echo "JUDGE FAILED"; exit 1; }
else
  echo "=== [$(date -Is)] SKIPPING judge (RUN_JUDGE=0). Long-form eval is judged, not"
  echo "    token-matched. Re-run with RUN_JUDGE=1, or use"
  echo "    judge-evals/scripts/verse_o_proj_in_judge.sh with ALGOS=atp-o_proj_in-respfix."
fi

echo "=== [$(date -Is)] DONE ==="
