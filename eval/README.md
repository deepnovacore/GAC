# GAC Evaluation Harness

This directory contains the checkpoint evaluation pipeline used to reproduce the numbers reported in the GAC paper (Tables 1–4). It covers **11 reported task slices from 9 dataset families across 4 domains**:

| Domain | Benchmarks | Script |
|---|---|---|
| Math | AMC · AIME24 · AIME25 | `math_bench.py` |
| Knowledge | MMLU-Pro · GPQA · SciBench | `knowledge_bench.py` |
| Code | MBPP · HumanEval | `code_bench.py` |
| Logic | BBH (Logical Deduction · Object Counting · Tracking) | `bbh_logic.py` |

## Design

**Generation** is unified via `generate_vllm.py` — a single vLLM-based
generator that takes a prompt file (jsonl), a system template, and produces
raw completions. This keeps sampling settings consistent across the reported
task slices (temperature = 0.6, top-p = 0.95, max_new_tokens = 8192 for the
reasoning setup).

**Scoring** uses domain-appropriate tooling:

| Domain | Scorer | Rationale |
|---|---|---|
| Math | [`math-verify`](https://github.com/huggingface/Math-Verify) | Canonical for AIME/AMC/MATH — the same scorer used by DeepSeek-R1, Qwen2.5-Math, LUFFY. Avoids false negatives from equivalent-but-formatted-differently answers (e.g. `\frac{1}{2}` vs `0.5`). |
| Code | [`bigcode-evaluation-harness`](https://github.com/bigcode-project/bigcode-evaluation-harness) + `sandbox_code_eval.sh` | Canonical pass@k execution in a Slurm CPU job with user/mount/network/PID isolation. The upstream executor alone is **not** a sandbox. |
| Multi-choice (MMLU-Pro, GPQA) | In-house letter matcher | Regex-extract `\boxed{}` / final "The answer is (X)" answer letter; strict A/B/C/D match. |
| Open-ended (SciBench) | In-house SymPy checker + numeric/string fallback | SciBench answers are numeric or symbolic expressions; no external judge is silently introduced. |
| BBH-Logic | In-house exact-match (case-insensitive, whitespace-normalized) | BBH answers are short strings ("(A)", "yes", "3") — exact match after normalization is standard. |

## Quick start

### 1 · Install

```bash
pip install -r requirements.txt
```

Additional deps for code eval:
```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness
cd bigcode-evaluation-harness && pip install -e .
```

### 2 · Run everything

```bash
bash run_all.sh \
    --model_path /path/to/gac-checkpoint-or-any-hf-model \
    --output_dir ./results \
    --tp_size 4 \
    --seed 0
```

Runs all reported task slices sequentially. For Qwen3.5, use a current vLLM
build that explicitly supports the Qwen3.5 architecture. On a multi-user
cluster, submit the command through the scheduler and let Slurm manage GPU
visibility; do not hand-edit `CUDA_VISIBLE_DEVICES`.

For a generic Slurm setup, export `MODEL_PATH` and `OUTPUT_DIR`, then submit
`eval/slurm_gpu_eval.sbatch`. After generation, export `BIGCODE_ROOT` and
`HF_HOME_DIR` and submit `eval/slurm_cpu_code_score.sbatch`. These examples
request resources through Slurm and keep code execution on a CPU allocation.

### 3 · Run individual benchmarks

```bash
# Math (AMC/AIME24/AIME25)
python math_bench.py \
    --model_path <model> \
    --benchmarks amc aime24 aime25 \
    --output_dir ./results \
    --tp_size 4

# Knowledge (MMLU-Pro/GPQA/SciBench)
python knowledge_bench.py \
    --model_path <model> \
    --benchmarks mmlu-pro gpqa scibench \
    --output_dir ./results \
    --tp_size 4

# Code generation (MBPP/HumanEval; no execution in this command)
python code_bench.py \
    --model_path <model> \
    --benchmarks mbpp humaneval \
    --output_dir ./results \
    --tp_size 4 \
    --n_samples 1     # pass@1

# Code scoring: run only inside the isolation wrapper, from a Slurm CPU job.
bash sandbox_code_eval.sh \
    --output_dir <results> \
    --bigcode_root /path/to/bigcode-evaluation-harness \
    --hf_home /path/to/hf_cache \
    --benchmarks mbpp humaneval \
    --mbpp_config full

# On a shared host, additionally hide each private data root, for example:
#   --hide_path /path/to/private/workspace

# Logic (BBH)
python bbh_logic.py \
    --model_path <model> \
    --benchmarks logical_deduction object_counting tracking_shuffled_objects \
    --output_dir ./results \
    --tp_size 4
```

## Reproducing paper numbers

The paper reports mean ± std over **3 seeds**. To reproduce:

```bash
for seed in 0 1 2; do
  bash run_all.sh --model_path <ckpt> --output_dir ./results/seed_${seed} --seed $seed
done
python aggregate.py ./results/seed_{0,1,2}   # prints mean±std across seeds
```

Expected numbers for `GAC + Token-φ` on Qwen2.5-7B:

| Benchmark | GAC + Token-φ (paper) |
|---|---|
| AMC | 67.2 ±0.4 |
| AIME24 | 20.8 ±0.4 |
| AIME25 | 19.8 ±0.5 |
| MMLU-Pro | 58.6 ±0.3 |
| MBPP | 78.8 ±0.5 |
| HumanEval | 83.5 ±0.4 |
| GPQA | 43.5 ±0.5 |
| SciBench | 41.2 ±0.5 |
| BBH-Logic (avg) | 65.7 ±0.5 |

## Data

Benchmark datasets are auto-downloaded from HuggingFace on first run and cached under `~/.cache/huggingface/`. Sources:

| Benchmark | HuggingFace dataset ID | Split |
|---|---|---|
| AMC | `AI-MO/aimo-validation-amc` | test (83 problems) |
| AIME24 | `HuggingFaceH4/aime_2024` | train (30 problems) |
| AIME25 | `math-ai/aime25` | test (30 problems) |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | test (12k problems, we use 1k-sample fixed subset) |
| GPQA | authors' public `gpqa_diamond.csv` (HF is gated) | 198 diamond problems |
| SciBench | `xw27/scibench` | train (692 problems in the cached release) |
| MBPP | `google-research-datasets/mbpp` | `full/test` (500 problems; `sanitized/test` has 257) |
| HumanEval | `openai/openai_humaneval` | test (164 problems) |
| BBH | `lukaemon/bbh` | 7 task configs × 250 test problems; three family averages |

Prompt templates are in `prompts.py`. The generator uses the checkpoint's own
chat template, so the same harness can evaluate Qwen2.5-family checkpoints and
Qwen3.5 checkpoints that are supported by the installed vLLM build.

## Output format

Each benchmark writes per-benchmark predictions and a summary:

```
results/
├── gpqa/predictions.jsonl
├── gpqa/summary.json
├── code/mbpp/generations.json
└── code/mbpp/summary.json
```

## Notes on scoring caveats

- **`math-verify`** returns `True/False/None`. `None` means the parser could not extract a candidate from the completion; we count these as incorrect (matching LUFFY / DeepSeek convention).
- **Code eval** requires execution of generated Python. `code_bench.py` never executes it; use `sandbox_code_eval.sh` from a Slurm CPU allocation. The wrapper requires offline HF cache access, disables network, hides common private roots plus any paths passed with `--hide_path`, and exposes only the exact evaluator/cache/output mounts. Do not replace it with a direct `--allow_code_execution` invocation.
- **SymPy fallback** on SciBench uses `sympy.simplify(gold - pred)`; a return of `0` means match. If SymPy raises, we fall back to string comparison after `.replace(' ', '')`.

## What this harness does NOT do

- **No training-time evaluation loop.** These scripts are for **checkpoint evaluation only** — you point them at a saved Hugging Face-compatible checkpoint or hub ID and get numbers.
- **No head-to-head baseline runs.** To reproduce the full paper table, you'd re-train HPT / LUFFY / CHORD / SRFT baselines yourself, or use the authors' released checkpoints. We only ship GAC.

Training scripts (VeRL fork) remain separate from this checkpoint-only
evaluator (see the main [README](../README.md#-roadmap)).

## Public-release hygiene

The evaluator writes raw prompts, gold answers, completions, and—during code
scoring—execution traces. These artifacts are intentionally not part of the
public Git repository or model repositories. Publish aggregate summaries only,
after checking dataset licenses and removing local absolute paths. The
evaluator itself records dataset sources and validates expected split sizes so
that a partial mirror cannot silently produce a headline number.

## License

Apache 2.0. This harness reuses code from LUFFY, `math-verify`, and `bigcode-evaluation-harness` — see individual files for attribution.
