# GAC Evaluation Harness

This directory contains the public checkpoint evaluation pipeline for GAC's
benchmark families and the Qwen3.5-4B release. It covers **11 reported task
slices from 9 dataset families across 4 domains**. Dataset access and sampling
are specified below; GPQA requires separate access.

| Domain | Benchmarks | Script |
|---|---|---|
| Math | AMC · AIME24 · AIME25 | `math_bench.py` |
| Knowledge | MMLU-Pro · GPQA · SciBench | `knowledge_bench.py` |
| Code | MBPP · HumanEval | `code_bench.py` |
| Logic | BBH (Logical Deduction · Object Counting · Tracking) | `bbh_logic.py` |

## Design

**Generation** is unified via `generate_vllm.py` — a single vLLM-based
generator that takes a prompt file (jsonl), a system template, and produces
raw completions. Reasoning-oriented slices use temperature = 0.6, top-p =
0.95, and max_new_tokens = 8192. Code generation deliberately uses a lower
temperature (0.2) and max_new_tokens = 1024; the exact settings are shown in
the model-card protocol table.

**Scoring** uses domain-appropriate tooling:

| Domain | Scorer | Rationale |
|---|---|---|
| Math | [`math-verify`](https://github.com/huggingface/Math-Verify) | Canonical for AIME/AMC/MATH — the same scorer used by DeepSeek-R1, Qwen2.5-Math, LUFFY. Avoids false negatives from equivalent-but-formatted-differently answers (e.g. `\frac{1}{2}` vs `0.5`). |
| Code | [`bigcode-evaluation-harness`](https://github.com/bigcode-project/bigcode-evaluation-harness) + `sandbox_code_eval.sh` | Canonical pass@k execution in a Slurm CPU job with user/mount/network/PID isolation. The upstream executor alone is **not** a sandbox. |
| Multi-choice (MMLU-Pro, GPQA) | In-house letter matcher | Regex-extract `\boxed{}` / final "The answer is (X)" answer letter; strict letter match (A–J for MMLU-Pro, A–D for GPQA). |
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

Runs all reported task slices sequentially. For Qwen3.5, use Transformers 5.10.4
or newer and a current vLLM build that explicitly supports the Qwen3.5
architecture. On a multi-user
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

Public benchmark datasets are downloaded from Hugging Face on first run.
Set `HF_HOME` to a data volume with sufficient space before downloading;
otherwise the default cache is `~/.cache/huggingface/`. GPQA additionally
requires dataset access or a user-provided CSV. Sources:

| Benchmark | HuggingFace dataset ID | Split |
|---|---|---|
| AMC | `AI-MO/aimo-validation-amc` | train (83 problems) |
| AIME24 | `HuggingFaceH4/aime_2024` | train (30 problems) |
| AIME25 | `math-ai/aime25` | test (30 problems) |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro`, revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` | `default/test`, 1,000 manifest-selected IDs from 12,032 problems |
| GPQA | gated `Idavidrein/gpqa` or a user-provided, lawfully obtained CSV | `gpqa_diamond/train`, all 198 problems; no GPQA data bundled |
| SciBench | `xw27/scibench` | train (692 problems in the cached release) |
| MBPP | `google-research-datasets/mbpp` | `full/test` (500 problems; `sanitized/test` has 257) |
| HumanEval | `openai/openai_humaneval` | test (164 problems) |
| BBH | `lukaemon/bbh` | 7 task configs × 250 test problems; three family averages |

### Fixed MMLU-Pro subset

The checked-in [manifest](manifests/mmlu_pro_test_b189ec765aa7ed75c8acfea42df31fdae71f97be.json)
records all 1,000 `question_id` values in evaluation order, their categories
and option counts, the dataset revision, and sampling metadata. It was
constructed with the earlier evaluator's algorithm: shuffle all test-row
indices with `random.Random(42)` and take the first 1,000. This is a fixed
pseudorandom sample, **not a stratified sample**. The generation `--seed`
does not change the subset.

The pinned full split contains questions with 3–10 options. The selected
subset has 827 questions with 10 options and 173 with 4–9 options. All selected
questions are evaluated with their actual options; none are padded or removed.
The loader validates option counts, category, unique IDs and answer-index
bounds. It resolves IDs from the manifest, so reordering dataset rows does
not change evaluation order. Summaries record the dataset revision and
manifest SHA-256; prediction IDs use `question_id` rather than row offsets.

This manifest freezes the public protocol for subsequent runs. Earlier runs
did not record a dataset revision, so the manifest by itself cannot establish
which revision those runs used. Regenerate predictions when validating an
earlier aggregate score against this frozen protocol.

Validate the download and all 1,000 selected rows on CPU before reserving GPUs
(requires `datasets`, but does not import vLLM or load a model):

```bash
python knowledge_bench.py --check_data --benchmarks mmlu-pro
```

### GPQA access

Obtain access through the [official GPQA dataset page](https://huggingface.co/datasets/Idavidrein/gpqa),
then authenticate with `hf auth login` (or set `HF_TOKEN` securely). The
evaluator uses that authentication through the Hugging Face datasets library.
Alternatively, pass a complete Diamond CSV that you have lawfully obtained:

```bash
python knowledge_bench.py --check_data --benchmarks gpqa \
    --gpqa_csv /path/to/gpqa_diamond.csv
```

The same `--gpqa_csv` option applies to evaluation, and `run_all.sh` accepts
the path through `GAC_GPQA_CSV`. Required columns are `Question`, `Correct Answer`,
and `Incorrect Answer 1`, `Incorrect Answer 2`, `Incorrect Answer 3`. The loader
requires 198 rows and records a CSV SHA-256 without exposing its local path.
No GPQA CSV or alternative data mirror is included in a clean checkout.

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
- With the default `n_samples=1`, all reported task scores are pass@1-style
  single-sample scores. If `n_samples > 1`, the non-code scorers use the best
  correct completion for each item; do not compare that setting to pass@1.
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
