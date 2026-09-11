#!/usr/bin/env bash
# Generic Slurm example for the CPU-only code scoring stage.
# Code execution must remain separate from GPU generation.
#SBATCH --job-name=gac-code-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=gac-code-score-%j.out
#SBATCH --error=gac-code-score-%j.err

set -euo pipefail

: "${OUTPUT_DIR:?Set OUTPUT_DIR to the generation results directory}"
: "${BIGCODE_ROOT:?Set BIGCODE_ROOT to a checked-out BigCode evaluator}"
: "${HF_HOME_DIR:?Set HF_HOME_DIR to a read-only dataset cache}"

EVAL_ROOT="${EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HIDE_ARGS=()
if [[ -n "${HIDE_PATH:-}" ]]; then
    HIDE_ARGS+=(--hide_path "$HIDE_PATH")
fi
read -r -a BENCHMARK_ARGS <<< "${BENCHMARKS:-mbpp humaneval}"

bash "$EVAL_ROOT/eval/sandbox_code_eval.sh" \
    --output_dir "$OUTPUT_DIR" \
    --bigcode_root "$BIGCODE_ROOT" \
    --hf_home "$HF_HOME_DIR" \
    --python_bin "$PYTHON_BIN" \
    --benchmarks "${BENCHMARK_ARGS[@]}" \
    --mbpp_config "${MBPP_CONFIG:-full}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-16}" \
    --timeout "${CODE_TIMEOUT:-3}" \
    "${HIDE_ARGS[@]}"
