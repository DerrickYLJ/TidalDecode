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
conda activate tidal

# Your original script begins here
MODEL_NAME="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
ATTN_TYPE="tidal"  # choices: tidal or None
CORRECTION_LAYER=13
TOP_K=128          # top-k
SPARSE_LAYER_START=2

echo "Running with --correction_layer ${CORRECTION_LAYER} ${TOP_K}"

python examples/run_tidal_llama.py \
    --model_name "${MODEL_NAME}" \
    --attn_type "${ATTN_TYPE}" \
    --top_k "${TOP_K}" \
    --correction_layer "${CORRECTION_LAYER}" \
    --sparse_layer_start "${SPARSE_LAYER_START}"

echo "Finished running with --correction_layer ${CORRECTION_LAYER}"
