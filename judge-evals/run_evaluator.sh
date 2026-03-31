#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH
cd /workspace/gcm-interp/judge-evals
exec python3 evaluator.py \
  --input_csv judge_prompts.csv \
  --judge \
  --batch_size 16 \
  --skip_rows 14656
