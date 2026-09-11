#!/usr/bin/env bash
# GAC eval harness — one-command reproduction across the reported task slices.
#
# Usage:
#   bash run_all.sh --model_path <ckpt> --output_dir ./results [--tp_size 4] [--seed 0]
#
# Runs, in order:
#   1. math_bench.py       (AMC / AIME24 / AIME25)
#   2. knowledge_bench.py  (MMLU-Pro / GPQA / SciBench)
#   3. code_bench.py       (MBPP / HumanEval)   [needs bigcode-evaluation-harness]
#   4. bbh_logic.py        (BBH logical subsets)
#
# On 4×A100/A800, wall-clock depends on model, vLLM build, and decoding length.
set -euo pipefail

MODEL_PATH=""
OUTPUT_DIR=""
TP_SIZE=4
SEED=0
N_SAMPLES=1
GPU_MEMORY_UTILIZATION=0.8
MBPP_CONFIG=full

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)  MODEL_PATH="$2"; shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --tp_size)     TP_SIZE="$2";    shift 2 ;;
        --seed)        SEED="$2";       shift 2 ;;
        --n_samples)   N_SAMPLES="$2";  shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --mbpp_config) MBPP_CONFIG="$2"; shift 2 ;;
        *)  echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_PATH" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --model_path <ckpt> --output_dir <dir> [--tp_size 4] [--seed 0] [--n_samples 1] [--gpu_memory_utilization 0.8] [--mbpp_config full|sanitized]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

KNOWLEDGE_EXTRA=()
if [[ -n "${GAC_GPQA_CSV:-}" ]]; then
    KNOWLEDGE_EXTRA+=(--gpqa_csv "$GAC_GPQA_CSV")
fi

# Keep-going is deliberate: each benchmark writes an independent summary, so
# one unavailable dataset must not hide results that already completed.
FAILURES=0

run_stage() {
    local label="$1"
    shift
    echo ""
    echo "$label"
    if ! "$@"; then
        echo "[warn] $label failed; continuing with remaining benchmarks" >&2
        FAILURES=$((FAILURES + 1))
    fi
}

echo "==============================================================="
echo "GAC eval harness"
echo "  model      : ${MODEL_PATH##*/}"
echo "  output_dir : $OUTPUT_DIR"
echo "  tp_size    : $TP_SIZE"
echo "  seed       : $SEED"
echo "  n_samples  : $N_SAMPLES"
echo "==============================================================="

run_stage "[1/4] Math (AMC / AIME24 / AIME25)" python math_bench.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tp_size "$TP_SIZE" \
    --n_samples "$N_SAMPLES" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --seed "$SEED"

run_stage "[2/4] Knowledge (MMLU-Pro / GPQA / SciBench)" python knowledge_bench.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tp_size "$TP_SIZE" \
    --n_samples "$N_SAMPLES" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --seed "$SEED" \
    "${KNOWLEDGE_EXTRA[@]}"

run_stage "[3/4] Code generation (MBPP / HumanEval; scoring is separate)" python code_bench.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tp_size "$TP_SIZE" \
    --n_samples "$N_SAMPLES" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --mbpp_config "$MBPP_CONFIG" \
    --seed "$SEED" || echo "[warn] code_bench failed — completions saved, install bigcode-evaluation-harness for scoring"

run_stage "[4/4] BBH-Logic (Logical Ded / Object Count / Tracking)" python bbh_logic.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tp_size "$TP_SIZE" \
    --n_samples "$N_SAMPLES" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --seed "$SEED"

echo ""
echo "==============================================================="
echo "Done. Per-benchmark summaries live under $OUTPUT_DIR/{math,knowledge,code,bbh_logic}_summary.json"
echo "Code pass@1 is produced separately by sandbox_code_eval.sh after generation."
echo "To aggregate across seeds:"
echo "  python aggregate.py $OUTPUT_DIR/../seed_{0,1,2}"
echo "==============================================================="

if [[ "$FAILURES" -gt 0 ]]; then
    echo "Completed with $FAILURES failed stage(s); inspect per-stage summaries." >&2
    exit 1
fi
