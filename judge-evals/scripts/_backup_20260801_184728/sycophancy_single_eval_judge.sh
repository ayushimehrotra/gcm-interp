#!/bin/bash
# Run the judge pipeline for the sycophancy task on single-eval
# for both Qwen and SOLAR, then generate w_rf and wo_rf heatmaps.

set -euo pipefail

cd /workspace/gcm-interp/judge-evals
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib/python3.11/dist-packages/torch/lib"

BATCH_SIZE=16

# Run judge pipeline on all sycophancy-single_eval gen files (both models)
python run_judge.py \
    --eval_subdir sycophancy-single_eval \
    --algos atp \
    --batch_size ${BATCH_SIZE} \
    --force

# Generate plots: sycophancy tasks, single eval, both rf modes
python /workspace/gcm-interp/plots.py \
    --tasks sycophancy-long sycophancy-single \
    --eval_variants single \
    --steer_variants long single \
    --rf_mode both

echo "Done."
