#!/bin/bash

mkdir -p results/aime

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ATTN_TYPE="tidal"         # or "tidal"
SPARSE_START_LAYER=2
DATA_ROOT="data/"
ATTENTION_SINK=4
MOST_RECENT_SCALE=2

for TEMPERATURE in 0.6 
do
    for CORRECTION_LAYER in 13
    do
        for TOP_K in 512
        do 
            echo "Running with --correction_layer ${CORRECTION_LAYER}, --sparse_layer_start ${SPARSE_START_LAYER}, --top_k ${TOP_K}, --temperature ${TEMPERATURE}, --attention_sink ${ATTENTION_SINK} --most_recent_scale ${MOST_RECENT_SCALE}"

            RESULT_FILE="results/aime/eval_aime_${TOP_K}_${ATTN_TYPE}_${CORRECTION_LAYER}_${TEMPERATURE}_${ATTENTION_SINK}_${MOST_RECENT_SCALE}.out"

            nohup python3 experiments/reasoning/aime_pass_1/eval_aime.py \
                --model_name "${MODEL_NAME}" \
                --attn_type "${ATTN_TYPE}" \
                --top_k "${TOP_K}" \
                --temperature "${TEMPERATURE}" \
                --correction_layer "${CORRECTION_LAYER}" \
                --sparse_layer_start "${SPARSE_START_LAYER}" \
                --attention_sink "${ATTENTION_SINK}" \
                --n 8 \
                --most_recent_scale_factor "${MOST_RECENT_SCALE}" \
                > "${RESULT_FILE}" 2>&1
        done
    done
done

echo "All runs submitted. Check 'results/aime' for output logs."
