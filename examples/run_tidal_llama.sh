#!/bin/bash

MODEL_NAME="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
ATTN_TYPE="tidal" # choices: tidal or None
CORRECTION_LAYER=13
TOP_K=128 # top-k
SPARSE_LAYER_START=2

echo "Running with --correction_layer ${CORRECTION_LAYER} ${TOP_K}"
  
python examples/run_tidal_llama.py \
    --model_name ${MODEL_NAME} \
    --attn_type ${ATTN_TYPE} \
    --top_k ${TOP_K} \
    --correction_layer ${CORRECTION_LAYER} \
    --sparse_layer_start ${SPARSE_LAYER_START} 

echo "Finished running with --correction_layer ${CORRECTION_LAYER}"