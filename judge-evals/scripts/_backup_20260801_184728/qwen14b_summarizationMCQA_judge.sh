#!/bin/bash
# Run the judge pipeline for the Qwen1.5-14B-Chat summarization-MCQA task
# (both single-eval and long-eval), i.e. the two transfer directories
#   results/Qwen1.5-14B-Chat/from_paragraphMCQA-long_to_sentenceMCQA-long
#   results/Qwen1.5-14B-Chat/from_paragraphMCQA-single_to_sentenceMCQA-single
#
# Task layout (same as the non-MCQA summarization task):
#   paragraphMCQA-single_eval -> MCQA over four candidate summaries; scored by letter
#                                matching (no judge model). Steering "succeeds" when the
#                                model picks the one-PARAGRAPH summary of the queried book
#                                instead of the one-sentence one the stem asks for.
#   paragraphMCQA-long_eval   -> free-form one-sentence summaries; scored by the
#                                "summarization" behavioral judge (Response (1) is longer
#                                than Response (2)) plus fluency + relevance.

set -euo pipefail

cd /home/ubuntu/gcm-interp/judge-evals
source /home/ubuntu/gcm-interp/.venv/bin/activate
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/ubuntu/gcm-interp/.venv/lib/python3.10/site-packages/nvidia/cu13/lib"
export VLLM_USE_DEEP_GEMM=0

BATCH_SIZE=16
MODEL=Qwen1.5-14B-Chat

# Single-eval: letter matching only (no judge model loaded)
python run_judge.py \
    --model_name ${MODEL} \
    --eval_subdir paragraphMCQA-single_eval \
    --algos atp \
    --batch_size ${BATCH_SIZE}

# Long-eval: behavioral judge + fluency + relevance.
# --no_judge_prefill drops the "(" seeded into the judge's assistant turn. The
# prefill made the judge hedge one step (rating 5 -> 4) without changing which
# responses it called longer, which deflated the ==5 pass rate; dropping it
# matches the reference pipeline's bare generation prompt.
python run_judge.py \
    --model_name ${MODEL} \
    --eval_subdir paragraphMCQA-long_eval \
    --algos atp \
    --batch_size ${BATCH_SIZE} \
    --no_judge_prefill

echo "Done."
