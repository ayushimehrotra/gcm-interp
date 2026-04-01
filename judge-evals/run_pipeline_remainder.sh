#!/bin/bash
set -e
cd /workspace/gcm-interp/judge-evals

echo "=== Waiting for judge (PID 3465) to finish ==="
while kill -0 3465 2>/dev/null; do sleep 10; done
echo "=== Judge done ==="

echo "=== Step 5: Fluency ==="
HF_TOKEN="your_token_here"
  --input_csv relevance_fluency_prompts.csv \
  --fluency \
  --batch_size 16
echo "=== Fluency done ==="

echo "=== Step 6: Relevance ==="
HF_TOKEN="your_token_here"
  --input_csv relevance_fluency_prompts.csv \
  --relevance \
  --batch_size 16
echo "=== Relevance done ==="

echo "=== Step 7: Compute accuracies ==="
python3 new-accuracies.py
echo "=== Accuracies done ==="

echo "=== Step 8: Plots ==="
cd /workspace/gcm-interp
python3 plots.py
echo "=== Plots done ==="
