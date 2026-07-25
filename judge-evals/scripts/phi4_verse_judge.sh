#!/bin/bash
# Run the judge pipeline for the phi-4 verse task (both single-eval and long-eval).

set -euo pipefail

cd /home/ubuntu/gcm-interp/judge-evals
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib"

BATCH_SIZE=16
MODEL=phi-4

# Single-eval: token matching only (no judge model loaded)
python run_judge.py \
    --model_name ${MODEL} \
    --eval_subdir verse-single_eval \
    --algos atp \
    --batch_size ${BATCH_SIZE}

# Long-eval: behavioral judge + fluency + relevance
python run_judge.py \
    --model_name ${MODEL} \
    --eval_subdir verse-long_eval \
    --algos atp \
    --batch_size ${BATCH_SIZE}

echo "Done."
