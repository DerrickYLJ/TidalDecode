#!/bin/bash

# Create a 'results/aime' directory to store logs
mkdir -p results/aime

# Model and default parameters
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ATTN_TYPE="None"         # or None if you don't want tidal attention
SPARSE_START_LAYER=2
DATA_ROOT="data/"

# You can loop over parameters as needed
for CORRECTION_LAYER in 13
do
    # You can set multiple top_k values here
    for TOP_K in 4096
    do 
        echo "Running with --correction_layer ${CORRECTION_LAYER}, --sparse_layer_start ${SPARSE_START_LAYER}, --top_k ${TOP_K}"

        # Log file to store output
        RESULT_FILE="results/aime/eval_aime_${TOP_K}_${ATTN_TYPE}_${CORRECTION_LAYER}.out"

        # Run the evaluation script in the background and redirect to RESULT_FILE
        nohup python3 experiments/reasoning/aime_pass_1/eval_aime.py \
            --model_name "${MODEL_NAME}" \
            --attn_type "${ATTN_TYPE}" \
            --top_k "${TOP_K}" \
            --correction_layer "${CORRECTION_LAYER}" \
            --sparse_layer_start "${SPARSE_START_LAYER}" \
            > "${RESULT_FILE}" 2>&1
    done
done

echo "All runs submitted. Check 'results/aime' for output logs."
