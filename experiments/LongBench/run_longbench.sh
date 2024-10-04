cd experiments/LongBench

export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p longbench_logs
model="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
attn_type="tidal"  # "tidal" or None

for task in "qmsum" "narrativeqa" "qasper" "multifieldqa_en" "triviaqa" "passage_retrieval_en" "hotpotqa" "2wikimqa" 
do 
    echo "Running task: $task with top_k: $topk"
    nohup python -u pred.py \
        --model $model --task $task \
        > "longbench_logs/${task}_topk${topk}_baseline_output.log"

    for topk in 4096
    do
        echo "Running task: $task with top_k: $topk"
        nohup python -u pred.py \
            --model $model --task $task \
            --attn_type $attn_type --top_k $topk  --chunk_size 16 \
            --sparse_layer_start 2 --correction_layer 13 \
            > "longbench_logs/${task}_${attn_type}_topk${topk}_output.log"
    done
done

python -u eval.py --model $model