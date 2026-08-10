#!/usr/bin/env bash
# Random-head control, ONE draw seed, ALL five models and both task families.
#
# Scope chosen over depth: breadth across models/tasks at seed 0 rather than
# 3 seeds on 2 models. NOTE this makes CLAUDE.md section 6.3 uncomputable -- the
# "is the random arm within noise of the real localization" test needs an
# across-seed spread, and one draw has none. random_control_analysis.py reports
# those conditions as "not yet checkable" rather than as a clean bill of health.
#
# Per model: the uniform arm generates only the -long localization and is
# mirrored into the -single one (the uniform draw ignores --source, so the two
# runs are bit-identical); the layer-matched arm generates every task, since its
# draw follows that localization's real per-layer ATP histogram.
#
# Models are ordered cheapest-first so results accumulate early; Qwen1.5-32B-Chat
# runs last. Falcon3 and Qwen1.5-14B-Chat are already complete at seed 0 and will
# resume through in a few minutes each.
#
# Resume is filename-based, so re-running is safe and skips finished conditions.

cd /home/ubuntu/gcm-interp
mkdir -p logs/random_control
PY=/home/ubuntu/gcm-interp/.venv/bin/python
SEED="${SEED:-0}"
MODELS="${MODELS:-falcon3 qwen olmo gemma qwen32b}"

declare -A MODEL_NAME=(
  [falcon3]="Falcon3-10B-Instruct"
  [qwen]="Qwen1.5-14B-Chat"
  [gemma]="gemma-3-12b-it"
  [olmo]="OLMo-2-1124-13B-DPO"
  [qwen32b]="Qwen1.5-32B-Chat"
)

run_one() {
  local s=$1 tag=$2 algos=$3
  echo "=== [$(date '+%F %T')] $tag"
  ALGOS="$algos" bash "$s" > "logs/random_control/${tag}.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    FAILED (rc=$rc) -- logs/random_control/${tag}.log"
    tail -5 "logs/random_control/${tag}.log" | sed 's/^/    | /'
  fi
  # A skipped cell is not a failure, but it must not pass unnoticed either.
  grep -h "^!! SKIPPING" "logs/random_control/${tag}.log" 2>/dev/null | sed 's/^/    /' | sort -u
}

for m in $MODELS; do
  echo "########## [$(date '+%F %T')] model: ${MODEL_NAME[$m]} (seed $SEED)"

  # uniform arm: -long localizations only, then mirror into the -single ones
  for task in vp summarization; do
    run_one "scripts/${m}_${task}-long_random.sh" "${m}_${task}-long__random-s${SEED}" "random-s${SEED}"
  done
  echo "=== [$(date '+%F %T')] mirror uniform arm: ${MODEL_NAME[$m]}"
  $PY random_control_mirror_uniform.py --model "${MODEL_NAME[$m]}" --seed "$SEED" --verify \
    >> "logs/random_control/mirror_${m}_s${SEED}.log" 2>&1 \
    || echo "    FAILED mirror ${MODEL_NAME[$m]} -- logs/random_control/mirror_${m}_s${SEED}.log"

  # layer-matched arm: every task on its own
  for task in vp summarization; do
    for loc in long single; do
      run_one "scripts/${m}_${task}-${loc}_random.sh" \
              "${m}_${task}-${loc}__randomlayer-s${SEED}" "randomlayer-s${SEED}"
    done
  done
done

echo "=== [$(date '+%F %T')] all models complete (seed $SEED)"
