#!/bin/bash
# Judge pipeline for the verse runs localized and steered on o_proj.INPUT
# (--patch_site o_proj_in), long-form eval only. Pick the model with MODEL=...:
#   MODEL=gemma-3-12b-it bash judge-evals/scripts/verse_o_proj_in_judge.sh
#
# Scores results/$MODEL/from_verse-{long,single}_to_prose/atp-o_proj_in/
# with judge + fluency + relevance (CLAUDE.md section 5, the w_rf number). That
# tree is separate from atp/, so nothing here can touch the published
# o_proj.output numbers.
#
# Settings pinned to the atp runs, or the comparison is void:
#   --no_judge_prefill   the '(' prefill shifts ratings down a step (5 -> 4) and
#                        w_rf counts only rating == 5, so omitting it yields
#                        plausible near-zero accuracies rather than an error.
#   --batch_size 16      BATCH_SIZE=16 in every committed judge script.
#
# One invocation per localization rather than a single pass:
# _evaluate_all_workdirs_batched() buffers a whole mode in memory and writes the
# JSONL only when that mode finishes, so a crash in a monolithic run discards
# hours of inference. Each chunk reloads the judge (~3-5 min); that buys
# restartability. Resume is accuracy-file based, so re-running is safe.
#
# NOTE run_judge.py exits 0 when it prepares nothing. Check
# "Phase 1 done: N files prepared" in the logs and count the accuracy files --
# never treat rc=0 as evidence of work.

set -uo pipefail
cd /home/ubuntu/gcm-interp/judge-evals

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib"
export VLLM_USE_DEEP_GEMM=0
# vLLM shells out to `ninja` during CUDA graph capture, so the venv's bin must be
# on PATH -- invoking .venv/bin/python directly is not enough.
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"

PY=/home/ubuntu/gcm-interp/.venv/bin/python
LOGDIR=/home/ubuntu/gcm-interp/logs/judge
mkdir -p "$LOGDIR"

MODEL="${MODEL:?set MODEL, e.g. MODEL=gemma-3-12b-it}"
ALGOS="${ALGOS:-atp-o_proj_in}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# "<source> <base> <eval_subdir>" -- both localizations, scored against the
# long-form eval set they were generated on.
CHUNKS=(
  "verse-long    prose  verse-long_eval"
  "verse-single  prose  verse-long_eval"
)

for chunk in "${CHUNKS[@]}"; do
  read -r src bas evalsub <<< "$chunk"
  tag="${MODEL}__from_${src}_to_${bas}__${evalsub}__o_proj_in"
  echo "=== [$(date '+%F %T')] $tag"
  $PY run_judge.py \
      --model_name "$MODEL" \
      --source "$src" \
      --base "$bas" \
      --eval_subdir "$evalsub" \
      --algos $ALGOS \
      --batch_size "$BATCH_SIZE" \
      --no_judge_prefill \
      > "$LOGDIR/${tag}.log" 2>&1
  rc=$?
  prepared=$(grep -o 'Phase 1 done: [0-9]* files prepared' "$LOGDIR/${tag}.log" | tail -1)
  if [[ $rc -ne 0 ]]; then
    echo "    FAILED (rc=$rc) -- $LOGDIR/${tag}.log"
    tail -15 "$LOGDIR/${tag}.log" | sed 's/^/    | /'
  else
    echo "    ok  ${prepared:-NO 'Phase 1 done' LINE -- SUSPECT}"
  fi
done

echo "=== [$(date '+%F %T')] all chunks complete"
find /home/ubuntu/gcm-interp/judge-evals/accuracy -path '*atp-o_proj_in*' -name '*accuracy.json' | wc -l \
  | xargs -I{} echo "accuracy files under atp-o_proj_in: {}  (expect 112 = 56 cells x wo_rf/w_rf x 2 localizations -> 224)"
