#!/bin/bash
# Judge pipeline for the random-head control arms, long-form evals only.
#
# Scores the *-long_eval side of the random-s* / randomlayer-s* trees with the
# judge + fluency + relevance metric (CLAUDE.md section 5, the w_rf number). The
# *-single_eval side is token-matched and was already scored; it is not touched
# here.
#
# Settings are pinned to whatever the atp runs used, or the arms are not
# comparable and the control is void:
#   --no_judge_prefill   every committed atp long-eval script passes it. The '('
#                        prefill shifts ratings down a step (5 -> 4).
#   --batch_size 16      BATCH_SIZE=16 in all committed judge scripts.
# Do not "optimise" either without re-running the atp side to match.
#
# Chunked one invocation per (model, task family) rather than a single --all
# pass. _evaluate_all_workdirs_batched() buffers an entire mode across every
# workdir in memory and writes the JSONL only after that mode finishes, so one
# monolithic run risks losing hours of inference to a single crash. Each chunk
# reloads the judge (~3-5 min) -- that overhead buys restartability.
#
# Resume is accuracy-file based (accuracy_exists in run_judge.py), so re-running
# is safe and finished conditions are skipped.
#
# Usage:
#   bash scripts/random_control_judge.sh                    # all four models
#   MODELS="gemma-3-12b-it" bash scripts/random_control_judge.sh
#   ALGOS="randomlayer-s0" bash scripts/random_control_judge.sh

set -uo pipefail
cd /home/ubuntu/gcm-interp/judge-evals

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib"
export VLLM_USE_DEEP_GEMM=0
# vLLM shells out to `ninja` when it JIT-compiles during CUDA graph capture, so
# the venv's bin must be on PATH -- calling .venv/bin/python directly is not
# enough and fails with FileNotFoundError: 'ninja' at engine start.
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"

PY=/home/ubuntu/gcm-interp/.venv/bin/python
LOGDIR=/home/ubuntu/gcm-interp/logs/judge
mkdir -p "$LOGDIR"

# Qwen1.5-32B-Chat is deliberately absent: its random arm is 16 of 448 files
# (N=1,2 only, uniform arm, verse-long only). The k-curve collapses N by max, so
# scoring a partial sweep would plot a misleadingly low point. Add it back once
# the regeneration has run.
MODELS="${MODELS:-Falcon3-10B-Instruct Qwen1.5-14B-Chat OLMo-2-1124-13B-DPO gemma-3-12b-it}"
ALGOS="${ALGOS:-random-s0 randomlayer-s0}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# One line per chunk: "<source> <base> <eval_subdir>".
# Both localizations of a family (-long and -single) are scored against that
# family's long-form eval set; the steering vector is already matched to the
# eval mode upstream, so there is nothing to cross here.
CHUNKS=(
  "verse-long      prose     verse-long_eval"
  "verse-single    prose     verse-long_eval"
  "paragraph-long  sentence  paragraph-long_eval"
  "paragraph-single sentence paragraph-long_eval"
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
    echo "    ok ($(grep -c 'ERROR' "$LOGDIR/${tag}.log" 2>/dev/null || echo 0) errors logged)"
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
