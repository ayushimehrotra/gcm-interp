#!/bin/bash
# Re-run the ablation/necessity dose analysis over the FULL grid.
#
# The first pass used necessity_dose.py's argument defaults, which are a 3-cell
# pilot (--models gemma-3-12b-it --tasks verse,summarization,bias) rather than
# the grid. It produced 30 rows / 3 cells where the superseded dose_*.csv files
# carry 210 rows / 21 cells. probe_units.py, contrastive_covariance.py and
# position_count_mechanism.py all default to the full grid, so only this one
# needs redoing.
#
# --ks matches the default 0.05,0.1,0.2,0.3,0.5, which is what the stale files
# used (210 rows = 21 cells x 2 arms x 5 budgets).
#
# Waits for the main rerun queue so only one job holds the GPU.
set -u
cd /home/ubuntu/gcm-interp/analysis
export PATH="/home/ubuntu/gcm-interp/.venv/bin:$PATH"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib
PY=/home/ubuntu/gcm-interp/.venv/bin/python

pat=$(printf 'rerun_stale_analys%s' 'es.sh')
while pgrep -f "$pat" >/dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %H:%M:%S')] main rerun queue clear, starting full-grid dose"

MODELS="gemma-3-12b-it,Qwen1.5-14B-Chat,Qwen1.5-32B-Chat,OLMo-2-1124-13B-DPO,Falcon3-10B-Instruct"
TASKS="verse,summarization,bias,factual recall,persona"

$PY necessity_dose.py --models "$MODELS" --tasks "$TASKS" \
    > /tmp/rerun_dose_fullgrid.log 2>&1
rc=$?
echo "[$(date '+%F %H:%M:%S')] full-grid dose rc=${rc}"
if [ "$rc" != "0" ]; then
  grep -iE 'error|traceback|out of memory' /tmp/rerun_dose_fullgrid.log | head -3 | sed 's/^/    /'
fi
echo "[$(date '+%F %H:%M:%S')] DOSE_FULLGRID_DONE"
