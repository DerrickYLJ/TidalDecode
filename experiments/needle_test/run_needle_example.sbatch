#!/bin/bash
#
# Just a typical Slurm header (adjust for your HPC’s requirements):
#SBATCH --job-name=tidal_llama
#SBATCH --output=tidal_llama_%j.out
#SBATCH --error=tidal_llama_%j.err
#SBATCH --gres=gpu:4           # request 1 GPU (adjust as needed)
#SBATCH --cpus-per-task=4      # number of CPU cores
#SBATCH --mem=64G              # total memory
#SBATCH --time=1:00:00         # time in hh:mm:ss

# (Optional) Load any modules you need for Python/CUDA
module load cuda-12.3

# (Optional) Activate your conda env if needed
source activate tidal

mkdir -p data
wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl -O ./data/pg19_mini.jsonl
# Create results directory if not exists
mkdir -p results/needle

MODEL_NAME="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
MAX_LENGTH=10000
MIN_LENGTH=1000
ROUNDS=5
ATTN_TYPE="tidal" # choices: tidal or None
OUTPUT_PATH="results/needle/"
RUN_NAME="TidalDecode"
CORRECTION_LAYER=13
TOP_K=128 # top-k
JOBS="14-15" # this will only run the last job (which is the maximum length)
SPARSE_LAYER_START=2
RESULT_FILE="results/needle/needle_${TOP_K}_${ATTN_TYPE}.out"

echo "Running with --correction_layer ${CORRECTION_LAYER} ${TOP_K}"
  
# Run the needle_test.py command with the current correction_layer
nohup python experiments/needle_test/needle_test.py \
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
    --sparse_layer_start ${SPARSE_LAYER_START} > ${RESULT_FILE}

echo "Finished running with --correction_layer ${CORRECTION_LAYER}" >> ${RESULT_FILE}