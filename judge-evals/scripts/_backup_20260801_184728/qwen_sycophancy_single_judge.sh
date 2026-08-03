#!/bin/bash

set -euo pipefail

cd /home/ubuntu/gcm-interp/judge-evals
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib/python3.11/dist-packages/torch/lib"
export HF_TOKEN="${HF_TOKEN:-}"

declare -a pairs=(
    "sycophancy-single_non-sycophantic"
    "sycophancy-long_non-sycophantic"
)

declare -a models=(
    "Qwen1.5-14B-Chat"
    "Qwen1.5-32B-Chat"
)

algos="atp"
BATCH_SIZE=16 
PLOTS=true 

for model_name in "${models[@]}"; do
    for pair in "${pairs[@]}"; do
        IFS='_' read -r source base <<< "$pair"

        cmd="python run_judge.py
            --model_name ${model_name}
            --source ${source}
            --base ${base}
            --algos ${algos}
            --batch_size ${BATCH_SIZE}"

        if [ "${PLOTS}" = true ]; then
            cmd="${cmd} --plots"
        fi

        eval ${cmd}
    done
done

echo ""
echo "All judge evaluations complete."