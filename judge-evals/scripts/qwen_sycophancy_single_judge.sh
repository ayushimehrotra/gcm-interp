#!/bin/bash
set -e
cd /workspace/gcm-interp/judge-evals
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH

# Step 1: Build merged eval outputs
if [ -f merged_eval_outputs.csv ]; then
    echo "=== Skipping one-giant-eval-file.py (merged_eval_outputs.csv exists) ==="
else
    echo "=== Running one-giant-eval-file.py ==="
    python3 one-giant-eval-file.py
    echo "=== Done ==="
fi

# Step 2: Build relevance/fluency prompts CSV
if [ -f relevance_fluency_prompts.csv ]; then
    echo "=== Skipping gen-relevance-fluency-dataframe.py (relevance_fluency_prompts.csv exists) ==="
else
    echo "=== Running gen-relevance-fluency-dataframe.py ==="
    HF_TOKEN=${HF_TOKEN} HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} python3 gen-relevance-fluency-dataframe.py
    echo "=== Done ==="
fi

# Step 3: Fluency eval (resume if partially done)
FLUENCY_OUT="relevance_fluency_prompts.fluency_prompt.judge_outputs.json"
FLUENCY_ACCURACY="relevance_fluency_prompts.fluency_prompt.judge_accuracy.json"
if [ -f "$FLUENCY_ACCURACY" ]; then
    echo "=== Skipping fluency eval ($FLUENCY_ACCURACY exists) ==="
else
    FLUENCY_DONE=0
    if [ -f "$FLUENCY_OUT" ]; then
        FLUENCY_DONE=$(wc -l < "$FLUENCY_OUT")
    fi
    echo "=== Running fluency eval (skip_rows=$FLUENCY_DONE) ==="
    python3 evaluator.py --input_csv relevance_fluency_prompts.csv --fluency --batch_size 16 --skip_rows $FLUENCY_DONE
    echo "=== Fluency done ==="
fi

# Step 4: Relevance eval (resume if partially done)
RELEVANCE_OUT="relevance_fluency_prompts.relevance_prompt.judge_outputs.json"
RELEVANCE_ACCURACY="relevance_fluency_prompts.relevance_prompt.judge_accuracy.json"
if [ -f "$RELEVANCE_ACCURACY" ]; then
    echo "=== Skipping relevance eval ($RELEVANCE_ACCURACY exists) ==="
else
    RELEVANCE_DONE=0
    if [ -f "$RELEVANCE_OUT" ]; then
        RELEVANCE_DONE=$(wc -l < "$RELEVANCE_OUT")
    fi
    echo "=== Running relevance eval (skip_rows=$RELEVANCE_DONE) ==="
    python3 evaluator.py --input_csv relevance_fluency_prompts.csv --relevance --batch_size 16 --skip_rows $RELEVANCE_DONE
    echo "=== Relevance done ==="
fi

# Step 5: Compute accuracies
echo "=== Computing accuracies ==="
python3 new-accuracies.py
echo "=== Accuracies done ==="

# Step 6: Generate plots
echo "=== Generating plots ==="
cd /workspace/gcm-interp
python3 plots.py
echo "=== Plots done ==="
