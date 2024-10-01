#!/bin/bash

### models:
# lmsys/longchat-7b-v1.5-32k
# gradientai/Llama-3-70B-Instruct-Gradient-1048k
# NousResearch/Yarn-Llama-2-7b-128k
# Parameters
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MAX_LENGTH=32000
MIN_LENGTH=2000
ROUNDS=5
ATTN_TYPE="index" # choices: index: ours; quest: quest
OUTPUT_PATH="./needle"
RUN_NAME="Index_s"
CORRECTION_LAYER=13
TOP_K=all # top-k
JOBS="14-15" # this will only run the last job (which is the maximum length)
SPARSE_LAYER_START=2
RESULT_FILE="results/needle/needle_${TOP_K}_${ATTN_TYPE}.out"

# Create results directory if not exists
mkdir -p results/needle

# Loop over correction_layer from 2 to 63
for TOP_K in {64,128,256,512,1024,2048}
do
  echo "Running with --correction_layer ${CORRECTION_LAYER} ${TOP_K}"
  
  # Run the needle_test.py command with the current correction_layer
  nohup python examples/needle_test/needle_test.py \
    --model_name ${MODEL_NAME} \
    --max_length ${MAX_LENGTH} \
    --min_length ${MIN_LENGTH} \
    --rounds ${ROUNDS} \
    --attn_type ${ATTN_TYPE} \
    --output_path ${OUTPUT_PATH} \
    --run_name ${RUN_NAME} \
    --top_k ${TOP_K} \
    --jobs ${JOBS} \
    --correction_layer ${CORRECTION_LAYER} \
    --sparse_layer_start ${SPARSE_LAYER_START} >> ${RESULT_FILE}

  echo "Finished running with --correction_layer ${CORRECTION_LAYER}" >> ${RESULT_FILE}
  echo "-------------------------------------------------------" >> ${RESULT_FILE}
done