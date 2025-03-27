#!/bin/bash
#
# Just a typical Slurm header (adjust for your HPC’s requirements):
#SBATCH --job-name=tidal_llama
#SBATCH --output=tidal_llama_%j.out
#SBATCH --error=tidal_llama_%j.err
#SBATCH --gres=gpu:4           # request 1 GPU (adjust as needed)
#SBATCH --cpus-per-task=4      # number of CPU cores
#SBATCH --mem=64G              # total memory
#SBATCH --time=30:00:00         # time in hh:mm:ss

# (Optional) Load any modules you need for Python/CUDA
module load cuda-12.3

# (Optional) Activate your conda env if needed
source activate tidal

mkdir -p results/aime

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ATTN_TYPE="tidal"
SPARSE_START_LAYER=2
DATA_ROOT="data/"

for TEMPERATURE in 0.6
do
    for CORRECTION_LAYER in 13
    do
        for TOP_K in 512
        do 
            echo "Running with --correction_layer ${CORRECTION_LAYER}, --sparse_layer_start ${SPARSE_START_LAYER}, --top_k ${TOP_K}, --temperature ${temperature}"

            RESULT_FILE="results/aime/eval_aime_${TOP_K}_${ATTN_TYPE}_${CORRECTION_LAYER}_${temperature}.out"

            nohup python3 experiments/reasoning/aime_pass_1/eval_aime.py \
                --model_name "${MODEL_NAME}" \
                --attn_type "${ATTN_TYPE}" \
                --top_k "${TOP_K}" \
                --n 8 \
                --temperature "${TEMPERATURE}" \
                --correction_layer "${CORRECTION_LAYER}" \
                --sparse_layer_start "${SPARSE_START_LAYER}" \
                > "${RESULT_FILE}" 2>&1
        done
    done
done

echo "All runs submitted. Check 'results/aime' for output logs."
