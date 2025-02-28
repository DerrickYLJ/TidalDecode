#!/bin/bash

mkdir -p results/aime

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ATTN_TYPE="None"         # or "tidal"
SPARSE_START_LAYER=2
DATA_ROOT="data/"

for TEMPERATURE in 0.1 0.2
do
    for CORRECTION_LAYER in 13
    do
        for TOP_K in 4096
        do 
            echo "Running with --correction_layer ${CORRECTION_LAYER}, --sparse_layer_start ${SPARSE_START_LAYER}, --top_k ${TOP_K}, --temperature ${temperature}"

            RESULT_FILE="results/aime/eval_aime_${TOP_K}_${ATTN_TYPE}_${CORRECTION_LAYER}_${temperature}.out"

            nohup python3 experiments/reasoning/aime_pass_1/eval_aime.py \
                --model_name "${MODEL_NAME}" \
                --attn_type "${ATTN_TYPE}" \
                --top_k "${TOP_K}" \
                --temperature "${TEMPERATURE}" \
                --correction_layer "${CORRECTION_LAYER}" \
                --sparse_layer_start "${SPARSE_START_LAYER}" \
                > "${RESULT_FILE}" 2>&1
        done
    done
done

echo "All runs submitted. Check 'results/aime' for output logs."
