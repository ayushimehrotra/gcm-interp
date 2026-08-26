#!/bin/bash
# Judge pipeline for the --localization_ctx 2x2 (Qwen1.5-14B-Chat / verse),
# long-form evals only.
#
# The *-single_eval side is token-matched, not judged: score it with
#   python compute_single_accuracies.py
# which scans the runs dir and picks up the new ctx trees on its own.
#
# Settings pinned to whatever the atp runs used, or the cells are not comparable:
#   --no_judge_prefill   the default '(' prefill shifts ratings down a step
#                        (5 -> 4) and w_rf counts only rating == 5, so omitting
#                        it yields plausible near-zero accuracies, not an error.
#   --batch_size 16      BATCH_SIZE=16 in every committed judge script.
#
# Chunked one invocation per (model, source) rather than a single --all pass:
# _evaluate_all_workdirs_batched() buffers a whole mode in memory and writes the
# JSONL only at the end, so a monolithic run risks losing hours to one crash.
#
# run_judge.py EXITS 0 WHEN IT PREPARES NOTHING. Never read rc as evidence of
# work -- check 'Phase 1 done: N files prepared' in the log and count the
# accuracy files.
#
# Usage:
#   bash scripts/ctx_sweep_judge.sh
#   ALGOS="atp-srcresp" bash scripts/ctx_sweep_judge.sh

set -uo pipefail
cd /home/ubuntu/gcm-interp/judge-evals

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib"
export VLLM_USE_DEEP_GEMM=0
# vLLM shells out to `ninja` during CUDA graph capture, so the venv's bin must be
# on PATH -- calling .venv/bin/python directly is not enough.
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"

PY=/home/ubuntu/gcm-interp/.venv/bin/python
LOGDIR=/home/ubuntu/gcm-interp/logs/judge_ctx
mkdir -p "$LOGDIR"

MODELS="${MODELS:-Qwen1.5-14B-Chat}"
ALGOS="${ALGOS:-atp-brsq atp-srcresp atp-baseq atp-baseq-srcresp}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# "<source> <base> <eval_subdir>". Both localizations are scored against the
# long-form eval set; the steering vector was already matched upstream.
CHUNKS=(
  "verse-long    prose  verse-long_eval"
  "verse-single  prose  verse-long_eval"
)

run_chunk() {
  local model=$1 source=$2 base=$3 evalsub=$4 tag=$5
  echo "=== [$(date '+%F %T')] $tag"
  $PY run_judge.py \
      --model_name "$model" \
      --source "$source" \
      --base "$base" \
      --eval_subdir "$evalsub" \
      --algos $ALGOS \
      --batch_size "$BATCH_SIZE" \
      --no_judge_prefill \
      > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    FAILED (rc=$rc) -- $LOGDIR/${tag}.log"
    tail -15 "$LOGDIR/${tag}.log" | sed 's/^/    | /'
  else
    grep -h "Phase 1 done" "$LOGDIR/${tag}.log" | sed 's/^/    /'
  fi
}

for model in $MODELS; do
  echo "########## [$(date '+%F %T')] model: $model"
  for chunk in "${CHUNKS[@]}"; do
    read -r src bas evalsub <<< "$chunk"
    run_chunk "$model" "$src" "$bas" "$evalsub" "${model}__from_${src}_to_${bas}__${evalsub}"
  done
done

echo "=== [$(date '+%F %T')] all chunks complete"
echo "accuracy files: /home/ubuntu/gcm-interp/judge-evals/accuracy"
