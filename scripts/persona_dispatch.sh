#!/bin/bash
# Dispatcher for the extraversion random-baseline runs.
#
#   * two model-jobs run concurrently
#   * launches are staggered 5 minutes apart
#   * Qwen1.5-32B runs ALONE -- it is the largest model, so it waits for the
#     pool to drain and nothing starts alongside it
#
# Each model-job is persona_random_one.sh, which runs that model's 8 invocations
# serially. Memory is not the constraint (all models load in 4-bit NF4: ~9-16 GB
# each, 98 GB total), so two concurrent jobs are comfortable.
#
# gemma-3-12b-it is already complete (448/448) and is omitted. Re-running any
# model is safe regardless: run.py skips generations that already exist.
set -u
cd /home/ubuntu/gcm-interp
LOGDIR=/tmp
STAGGER="${STAGGER:-300}"      # 5 minutes between launches
MAXJOBS="${MAXJOBS:-2}"

# model_id|model_name|solo   (solo=1 -> must run with nothing else)
QUEUE=(
  "allenai/OLMo-2-1124-13B-DPO|OLMo-2-1124-13B-DPO|0"
  "Qwen/Qwen1.5-14B-Chat|Qwen1.5-14B-Chat|0"
  "tiiuae/Falcon3-10B-Instruct|Falcon3-10B-Instruct|0"
  "Qwen/Qwen1.5-32B-Chat|Qwen1.5-32B-Chat|1"
)

PIDS=()
running() {
  local n=0 p
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && n=$((n+1)); done
  echo $n
}

for entry in "${QUEUE[@]}"; do
  IFS='|' read -r mid mname solo <<< "$entry"
  if [ "$solo" = "1" ]; then
    while [ "$(running)" -gt 0 ]; do sleep 60; done      # drain the pool
    echo "[$(date '+%F %H:%M:%S')] DISPATCH $mname (solo)"
  else
    while [ "$(running)" -ge "$MAXJOBS" ]; do sleep 60; done
    echo "[$(date '+%F %H:%M:%S')] DISPATCH $mname"
  fi
  bash scripts/persona_random_one.sh "$mid" "$mname" \
       >> "$LOGDIR/persona_${mname}.log" 2>&1 &
  PIDS+=($!)
  sleep "$STAGGER"
done

for p in "${PIDS[@]:-}"; do [ -n "$p" ] && wait "$p" 2>/dev/null; done
echo "[$(date '+%F %H:%M:%S')] PERSONA_DISPATCH_DONE"
