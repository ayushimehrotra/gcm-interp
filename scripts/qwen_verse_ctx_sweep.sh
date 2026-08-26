#!/usr/bin/env bash
# --localization_ctx 2x2 for Qwen1.5-14B-Chat / verse, both localizations.
#
# Varies which sequences supply the two activations ATP differences
# (eval/localization_ctx.py):
#
#   br-sq   A_src = source prompt only        A_base_patch = base prompt+response
#   br-sr   A_src = source prompt+response    A_base_patch = base prompt+response
#   bq-sq   A_src = source prompt only        A_base_patch = base prompt only
#   bq-sr   A_src = source prompt+response    A_base_patch = base prompt only
#
# The scored/differentiated base run always carries its response, so the gradient
# factor is identical across all four -- only the subtrahend and the source move.
#
# br-sq is the current pairing addressed EXPLICITLY, so it lands in atp-brsq/
# rather than atp/. That is deliberate: atp/ holds generations produced on
# another machine, resume here is filename-based, and section 2 of CLAUDE.md
# shows greedy decoding diverges between machines. Regenerating the control
# locally keeps all four cells on one provenance.
#
# Sweeps are pinned to this model's existing atp sweep or the cells are not
# comparable (CLAUDE.md section 4).
#
# Usage:
#   bash scripts/qwen_verse_ctx_sweep.sh
#   CTXS="br-sr" LOCS="verse-long" bash scripts/qwen_verse_ctx_sweep.sh

set -uo pipefail
export RM_INTERP_REPO="/home/ubuntu/gcm-interp"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib"
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"
cd "$RM_INTERP_REPO"

PY=/home/ubuntu/gcm-interp/.venv/bin/python
model_id="Qwen/Qwen1.5-14B-Chat"
model_name="Qwen1.5-14B-Chat"
device="cuda:0"
base="prose"

CTXS="${CTXS:-br-sq br-sr bq-sq bq-sr}"
LOCS="${LOCS:-verse-long verse-single}"
STEERING_FACTORS="${STEERING_FACTORS:-1,2,4,5,6,8,10}"
TOPK_VALS="${TOPK_VALS:-0.01,0.03,0.05,0.07,0.09,0.1,0.5,1.0}"

# Site and span. Defaults reproduce the committed atp sweep (o_proj_out/legacy).
# Set PATCH_SITE=o_proj_in RESPONSE_SPAN=full for the corrected grid: at
# o_proj_in block u IS head u (at o_proj_out it is the 128 residual coords a head
# writes into, after W_O has mixed them), and `full` scores the FIRST response
# token -- which for -single data is the entire answer, so under `legacy` the
# single-token localization never scores the letter it is supposed to select for.
# Suffixes compose as atp-o_proj_in-respfix-<ctx>; judge-evals strips them back.
PATCH_SITE="${PATCH_SITE:-o_proj_out}"
RESPONSE_SPAN="${RESPONSE_SPAN:-legacy}"
echo "site=$PATCH_SITE span=$RESPONSE_SPAN ctxs=$CTXS locs=$LOCS"

LOGDIR="$RM_INTERP_REPO/logs/ctx_sweep"
mkdir -p "$LOGDIR"

run_step() {
  local tag=$1; shift
  echo "=== [$(date '+%F %T')] $tag"
  "$@" > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    FAILED (rc=$rc) -- $LOGDIR/${tag}.log"
    tail -8 "$LOGDIR/${tag}.log" | sed 's/^/    | /'
  fi
  return $rc
}

for loc in $LOCS; do
  for ctx in $CTXS; do
    echo "########## [$(date '+%F %T')] $loc / ctx=$ctx"

    run_step "${loc}__${ctx}__patch" \
      $PY run.py --model_id "$model_id" --batch_size 1 --patch_algo atp \
                 --source "$loc" --base "$base" --device "$device" \
                 --localization_ctx "$ctx" --patch_site "$PATCH_SITE" \
                 --response_span "$RESPONSE_SPAN" --patch_model || continue

    # long-form eval, long-form steering vector (matched; CLAUDE.md section 4)
    run_step "${loc}__${ctx}__eval-long" \
      $PY run.py --model_id "$model_id" --batch_size 1 --patch_algo atp \
                 --source "$loc" --base "$base" --device "$device" \
                 --localization_ctx "$ctx" --patch_site "$PATCH_SITE" \
                 --response_span "$RESPONSE_SPAN" \
                 --eval_model --kv_caching --steering --ablation steer \
                 --topk_vals "$TOPK_VALS" --steering_factors "$STEERING_FACTORS" \
                 --eval_test "$RM_INTERP_REPO/data/${model_name}/verse-long/prose-test.jsonl" \
                 --steering_add_path "$RM_INTERP_REPO/data/${model_name}/verse-long/verse-long-desired-all.jsonl" \
                 --steering_sub_path "$RM_INTERP_REPO/data/${model_name}/verse-long/prose-desired-all.jsonl"

    # single-token eval, single-token steering vector (matched)
    run_step "${loc}__${ctx}__eval-single" \
      $PY run.py --model_id "$model_id" --batch_size 1 --patch_algo atp \
                 --source "$loc" --base "$base" --device "$device" \
                 --localization_ctx "$ctx" --patch_site "$PATCH_SITE" \
                 --response_span "$RESPONSE_SPAN" \
                 --eval_model --kv_caching --steering --ablation steer \
                 --topk_vals "$TOPK_VALS" --steering_factors "$STEERING_FACTORS" \
                 --eval_test "$RM_INTERP_REPO/data/${model_name}/verse-single/prose-test.jsonl" \
                 --steering_add_path "$RM_INTERP_REPO/data/${model_name}/verse-single/verse-single-steering.jsonl" \
                 --steering_sub_path "$RM_INTERP_REPO/data/${model_name}/verse-single/prose-single-steering.jsonl"
  done
done

echo "=== [$(date '+%F %T')] ctx sweep complete"
