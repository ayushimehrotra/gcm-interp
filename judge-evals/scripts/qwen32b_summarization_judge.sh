#!/bin/bash
# Run the judge pipeline for the Qwen1.5-32B-Chat summarization task
# (both single-eval and long-eval).
#
# Task layout:
#   paragraph-single_eval -> MCQA over four candidate summaries; scored by letter
#                            matching (no judge model). Steering "succeeds" when the
#                            model picks the one-PARAGRAPH summary of the queried book
#                            instead of the one-sentence one the stem asks for.
#   paragraph-long_eval   -> free-form one-sentence summaries; scored by the
#                            "summarization" behavioral judge (Response (1) is longer
#                            than Response (2)) plus fluency + relevance.

set -euo pipefail

cd /home/ubuntu/gcm-interp/judge-evals
source /home/ubuntu/gcm-interp/.venv/bin/activate
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib"
export VLLM_USE_DEEP_GEMM=0

BATCH_SIZE=16
MODEL=Qwen1.5-32B-Chat

# Single-eval: letter matching only (no judge model loaded)
python run_judge.py \
    --model_name ${MODEL} \
    --eval_subdir paragraph-single_eval \
    --algos atp \
    --batch_size ${BATCH_SIZE}

# Long-eval: behavioral judge + fluency + relevance
python run_judge.py \
    --model_name ${MODEL} \
    --eval_subdir paragraph-long_eval \
    --algos atp \
    --batch_size ${BATCH_SIZE} \
    --no_judge_prefill

echo "Done."
