#!/bin/bash
set -e
cd /workspace/gcm-interp/judge-evals

echo "=== Waiting for fluency job (PID 4676) ==="
while kill -0 4676 2>/dev/null; do sleep 30; done
echo "=== Fluency done ==="

echo "=== Step 6: Relevance ==="
python3 evaluator.py \
  --input_csv relevance_fluency_prompts.csv \
  --relevance \
  --batch_size 16
echo "=== Relevance done ==="

echo "=== Step 7: Compute accuracies ==="
python3 new-accuracies.py
echo "=== Accuracies done ==="

echo "=== Plots ==="
cd /workspace/gcm-interp
python3 plots.py
echo "=== Plots done ==="
