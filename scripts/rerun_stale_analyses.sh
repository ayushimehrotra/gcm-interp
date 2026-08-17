#!/bin/bash
# Re-run the analyses whose CSVs predate the 2026-08-13 repo pull.
#
# The pull restored umang's results_with_answers / results_pipeline_with_answers
# trees, which changed which cells are available. Everything dated Aug 12 was
# computed against the older, thinner tree and is not consistent with the
# current figures. These four scripts all need model forward passes, so they
# were blocked while the 70B judge held the GPU.
#
# Serial on purpose: each loads a model at 4-bit and the GPU is one card.
set -u
cd /home/ubuntu/gcm-interp/analysis
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
PY=/home/ubuntu/gcm-interp/.venv/bin/python

run () {
  local name="$1"; shift
  echo "[$(date '+%F %H:%M:%S')] START $name"
  "$@" > "/tmp/rerun_${name}.log" 2>&1
  local rc=$?          # captured immediately; a $(...) in the echo would reset it
  echo "[$(date '+%F %H:%M:%S')] END   $name rc=${rc}"
  if [ "$rc" != "0" ]; then
    echo "    FAILED -> /tmp/rerun_${name}.log"
    grep -iE 'error|traceback|out of memory' "/tmp/rerun_${name}.log" | head -3 | sed 's/^/      /'
  fi
}

run probe        $PY probe_units.py
run dose         $PY necessity_dose.py
run covariance   $PY contrastive_covariance.py
run poscount_ay  $PY position_count_mechanism.py --repo ayushi --out position_count.csv
run poscount_um  $PY position_count_mechanism.py --repo umang  --out position_count_umang.csv

echo "[$(date '+%F %H:%M:%S')] RERUN_STALE_DONE"
