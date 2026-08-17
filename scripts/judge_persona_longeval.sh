#!/bin/bash
# Free-form (long-eval) judging for the persona random control arms, seed 0.
#
# This is the corrected replacement for run_judge_persona_all.sh, which reported
# rc=0 on all 20 cells while producing nothing. Four bugs were in play:
#
#   1. 'extraversion' was missing from SOURCE_TO_TEMPLATE, so every free-form
#      cell died with "unknown SOURCE" (the template itself already existed).
#      Fixed in judge-evals/config.py, plus PAIRED_TEMPLATES membership --
#      umang builds this task as Response (1)=steered / (2)=unsteered.
#   2. --no_judge_prefill was not passed. The default "(" prefill shifts ratings
#      down one step (5 -> 4) and w_rf counts only rating == 5, so every accuracy
#      would have collapsed toward 0. umang's pipeline uses no prefill.
#   3. The single-token cells resolved their test set from the localization name
#      instead of the eval mode. Fixed by resolve_test_base() in
#      compute_single_accuracies.py; those cells are already recomputed.
#   4. compute_single_accuracies scores an unconditional yes-rate where umang
#      scores a flip rate conditioned on the unsteered answer not being "yes".
#      Numerically identical here -- the unsteered baseline never answers "yes"
#      in single-token eval (measured 0.000 across all 5 models, both trees) --
#      so no single-token number changes. Recorded because it would bite on any
#      task whose baseline is not at floor.
#
# Only the 10 free-form cells run here; the 10 single-token cells are done and
# correct. --force is NOT passed, so anything already scored is skipped.
#
# BATCH_SIZE 16 and no-prefill are fixed by CLAUDE.md section 5 -- changing
# either voids the comparison against the atp arms.
set -u
cd /home/ubuntu/gcm-interp/judge-evals
export VLLM_USE_DEEP_GEMM=0
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"   # vLLM shells out to ninja
PY=/home/ubuntu/gcm-interp/.venv/bin/python

for model in gemma-3-12b-it OLMo-2-1124-13B-DPO Qwen1.5-14B-Chat \
             Falcon3-10B-Instruct Qwen1.5-32B-Chat; do
  for loc in long single; do
    tag="${model}__extraversion-${loc}__eval-long"
    echo "[$(date '+%F %H:%M:%S')] JUDGE $tag"
    $PY run_judge.py \
        --model_name "$model" \
        --source "extraversion-${loc}" \
        --base "introversion-${loc}" \
        --eval_subdir "extraversion-long_eval" \
        --algos random-s0 randomlayer-s0 \
        --no_judge_prefill \
        --batch_size 16 \
        > "/tmp/jlong_${tag}.log" 2>&1
    rc=$?     # captured on its own line; a $(...) in the echo would reset it
    n=$(find "/home/ubuntu/gcm-interp/judge-evals/accuracy/${model}/from_extraversion-${loc}_to_introversion-${loc}" \
             -path '*extraversion-long_eval*' -name '*w_rf*accuracy.json' 2>/dev/null | wc -l)
    echo "[$(date '+%F %H:%M:%S')] DONE  $tag  rc=${rc}  w_rf_files=${n}"
    if [ "$rc" != "0" ] || [ "$n" = "0" ]; then
      echo "    PROBLEM -> /tmp/jlong_${tag}.log"
      grep -iE 'error|traceback' "/tmp/jlong_${tag}.log" | head -3 | sed 's/^/      /'
    fi
  done
done
echo "[$(date '+%F %H:%M:%S')] JUDGE_PERSONA_LONGEVAL_DONE"
