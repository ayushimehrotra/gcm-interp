#!/bin/bash
set -e
cd /home/ubuntu/gcm-interp/judge-evals

export VLLM_USE_DEEP_GEMM=0

# # Long-form localization (sycophancy-long source, long-eval test set)
# .venv/bin/python run_judge.py \
#     --model_name Qwen1.5-32B-Chat \
#     --source sycophancy-long \
#     --base non-sycophantic \
#     --eval_subdir sycophancy-long_eval \
#     --algos atp \
#     --force

# # Single-token localization (sycophancy-single source, long-eval test set)
# .venv/bin/python run_judge.py \
#     --model_name Qwen1.5-32B-Chat \
#     --source sycophancy-single \
#     --base non-sycophantic \
#     --eval_subdir sycophancy-long_eval \
#     --algos atp \
#     --force

# Long-form localization (sycophancy-long source, single-eval test set — token matching only)
.venv/bin/python run_judge.py \
    --model_name Qwen1.5-32B-Chat \
    --source sycophancy-long \
    --base non-sycophantic \
    --eval_subdir sycophancy-single_eval \
    --algos atp \
    --skip_judge \
    --force

# Single-token localization (sycophancy-single source, single-eval test set — token matching only)
.venv/bin/python run_judge.py \
    --model_name Qwen1.5-32B-Chat \
    --source sycophancy-single \
    --base non-sycophantic \
    --eval_subdir sycophancy-single_eval \
    --algos atp \
    --skip_judge \
    --force
