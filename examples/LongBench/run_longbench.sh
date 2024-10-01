cd examples/LongBench

export CUDA_VISIBLE_DEVICES=0,1,2,3

model="meta-llama/Llama-3.1-8B-Instruct"
attn_type="index"  # Define quest as a parameter

for task in "qmsum" "narrativeqa" "triviaqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "passage_retrieval_en"
do 
    echo "Running task: $task with top_k: $topk"
    nohup python -u pred.py \
        --model $model --task $task \
        > "${task}_topk${topk}_baseline_output_NEW.log"

    for topk in 1024 4096
    do
        echo "Running task: $task with top_k: $topk"
        nohup python -u pred.py \
            --model $model --task $task \
            --attn_type $attn_type --top_k $topk  --chunk_size 16 \
            --sparse_layer_start 2 --correction_layer 13 \
            > "${task}_${attn_type}_topk${topk}_output_NEW.log"
    done
done