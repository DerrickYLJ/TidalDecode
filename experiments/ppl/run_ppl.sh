MODELPATH=gradientai/Llama-3-8B-Instruct-Gradient-1048k
EVAL_TOKEN=32000
TOP_K=4096
OUTPUT_DIR=results/ppl/${TOP_K}_${EVAL_TOKEN}/
mkdir -p $OUTPUT_DIR

nohup python -u experiments/ppl/ppl.py \
    --model_name_or_path $MODELPATH \
    --output_dir $OUTPUT_DIR \
    --num_eval_tokens $EVAL_TOKEN \
    --correction_layer 13 \
    --sparse_layer_start 2 \
    --attn_type "tidal" --top_k $TOP_K  > ${OUTPUT_DIR}/output_${EVAL_TOKEN}_${TOP_K}.log