#!/bin/bash
# Run the judge pipeline for the OLMo verse task (both single-eval and long-eval).

set -euo pipefail

cd /home/ubuntu/gcm-interp/judge-evals
source /home/ubuntu/.venv/bin/activate
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib/python3.11/dist-packages/torch/lib:/home/ubuntu/cuda-compat:/home/ubuntu/.venv/lib/python3.10/site-packages/nvidia/cu13/lib"
export LD_PRELOAD="/home/ubuntu/.venv/lib/python3.10/site-packages/nvidia/cu13/lib/libnvJitLink.so.13"

BATCH_SIZE=16
MODEL=OLMo-2-1124-13B-DPO

# Single-eval: token matching only (no judge model loaded)
python run_judge.py \
    --model_name ${MODEL} \
    --eval_subdir sycophancy-single_eval \
    --algos atp \
    --batch_size ${BATCH_SIZE}
    

# Long-eval: behavioral judge + fluency + relevance
python run_judge.py \
    --model_name ${MODEL} \
    --eval_subdir sycophancy-long_eval \
    --algos atp \
    --batch_size ${BATCH_SIZE} \

echo "Done."
