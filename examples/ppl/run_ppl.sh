MODELPATH=gradientai/Llama-3-8B-Instruct-Gradient-1048k
OUTPUT_DIR=results/ppl/4096_100K/
mkdir -p $OUTPUT_DIR

budget=4096

nohup python -u examples/ppl/run_ppl.py \
    --model_name_or_path $MODELPATH \
    --output_dir $OUTPUT_DIR \
    --num_eval_tokens 96000 \
    --correction_layer 13 \
    --sparse_layer_start 2 \
    --attn_type index --top_k $budget  --chunk_size 16 > output_100K_4096.log