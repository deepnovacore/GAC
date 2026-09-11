#!/usr/bin/env bash
# Generic Slurm example for the GPU generation stages.
# Set MODEL_PATH and OUTPUT_DIR in the environment before submitting.
# Slurm owns GPU allocation and CUDA visibility; this script does not override
# CUDA_VISIBLE_DEVICES.
#SBATCH --job-name=gac-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=gac-eval-%j.out
#SBATCH --error=gac-eval-%j.err

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local checkpoint or hub model ID}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a dedicated results directory}"

EVAL_ROOT="${EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TP_SIZE="${TP_SIZE:-${SLURM_GPUS_ON_NODE:-4}}"

"$PYTHON_BIN" --version
bash "$EVAL_ROOT/eval/run_all.sh" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tp_size "$TP_SIZE" \
    --seed "${SEED:-0}" \
    --n_samples "${N_SAMPLES:-1}"
