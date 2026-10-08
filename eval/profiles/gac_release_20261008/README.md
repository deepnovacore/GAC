# GAC-Qwen3.5-4B · October 2026 release profile

This profile reproduces the **protocol** of the October release table:
one checkpoint, decoding seeds 0/1/2, and a mean over all three seeds.
It is separate from the original paper experiments and the general
evaluator's task-specific defaults. It does not select a best seed.

## Frozen configuration

| Setting | Value |
|:--|:--|
| Task slices / questions per seed | 11 / 4,447 |
| Sampling | temperature 0.6, top-p 0.95, one answer per question |
| Output / model context | 8,192 / 16,384 tokens |
| Inference | BF16, tensor parallelism 4, memory utilization 0.8 |
| Seeds | engine seed 0; decoding seeds 0, 1, 2 |
| Batching | One joint batch of every slice per seed |
| Sampler | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| Engine | vLLM 0.23.0; prefix cache off, chunked prefill on |
| Code executor | BigCode `8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd`, 8 workers, 3 seconds/program |
| Math equivalence | `math-verify==0.9.0` |

Software, hardware and batching can affect sampled outputs even with fixed
seeds. The generation manifest records installed package versions. No script
changes a score to match a published value. Aggregation refuses partial or
duplicate task/seed groups.

## Data

| Slice | Dataset / config / split | N |
|:--|:--|--:|
| AMC | `AI-MO/aimo-validation-amc` / default / train | 83 |
| AIME24 | `HuggingFaceH4/aime_2024` / default / train | 30 |
| AIME25 | `math-ai/aime25` / default / test | 30 |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` / default / test; pinned 1,000-ID manifest | 1,000 |
| GPQA-Diamond | `Idavidrein/gpqa` / gpqa_diamond / train | 198 |
| SciBench | `xw27/scibench` / default / train | 692 |
| MBPP | `google-research-datasets/mbpp` / full / test | 500 |
| HumanEval | `openai/openai_humaneval` / default / test | 164 |
| BBH Logical Deduction | `lukaemon/bbh` / three-, five-, seven-object logical deduction / test | 750 |
| BBH Object Counting | `lukaemon/bbh` / object_counting / test | 250 |
| BBH Tracking | `lukaemon/bbh` / three-, five-, seven-object shuffled tracking / test | 750 |

MMLU-Pro is pinned to revision
`b189ec765aa7ed75c8acfea42df31fdae71f97be`; its seed-42 sample is pseudorandom,
not stratified. Questions are read in checked-in manifest order and actual
option counts are preserved. GPQA option permutations use the row index,
independently of the decoding seed. Obtain GPQA access and authenticate with
HF, or pass a lawfully obtained, complete 198-row CSV via `--gpqa_csv`.
**No GPQA questions or answers are bundled.**

An ordered, identifier-neutral fingerprint covers task names, prompts, gold
answers and original tests. `prepare.py` refuses a different dataset/order:
`b4e4095a62395764927efe2b7b04992b1018895f81532b6b909dfefc62ebee8e`.
Other sources are validated by this content fingerprint rather than a claim
that every upstream repository has an immutable revision pin.

## Run

From the repository root, install the generation/scoring dependencies:

```bash
pip install -r eval/profiles/gac_release_20261008/requirements.txt
python eval/profiles/gac_release_20261008/prepare.py --output_dir ./release_eval
```

If needed, add `--gpqa_csv /path/to/gpqa_diamond.csv`. The output directory
contains evaluation questions, original tests and predictions: keep it private
and follow each dataset's access/license conditions. Existing artifacts are
never overwritten; use a new output directory for another run.

Launch the following **inside a scheduler-allocated four-GPU job**, not on a
Slurm login node. Do not manually bind or share another job's GPUs:

```bash
python eval/profiles/gac_release_20261008/generate.py \
  --model_path YueLinHu/GAC-Qwen3.5-4B --model_revision release-20261008 \
  --output_dir ./release_eval --tp_size 4
```

For Slurm, use `sbatch --nodes=1 --ntasks=1 --gres=gpu:4 ...` with your
cluster's partition, CPU/memory and time settings. All 11 slices run in one
batch for each seed. Generation never executes generated code.

On a **CPU evaluation job**, prepare scores and execute original code tests
through the fail-closed Linux namespace wrapper:

```bash
python eval/profiles/gac_release_20261008/score.py prepare --output_dir ./release_eval
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git ./bigcode-harness
git -C ./bigcode-harness checkout 8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd
# Install the upstream executor dependencies in a trusted CPU environment.
bash eval/profiles/gac_release_20261008/sandbox_code_eval.sh \
  --output_dir ./release_eval --bigcode_root ./bigcode-harness \
  --python_bin /path/to/cpu-environment/bin/python
python eval/profiles/gac_release_20261008/score.py finalize --output_dir ./release_eval
```

The optional `--extra_pythonpath` exposes one exact dependency directory
read-only. The wrapper requires unprivileged user/mount/network/PID namespaces
and chroot/setpriv support; it drops capabilities before code execution and
never falls back to unsandboxed execution. Only the
selected evaluation output directory is a writable host mount. Host homes,
credentials, workspace data and GPUs are not exposed. Defense in depth is not
a guarantee against malicious kernel exploits; use a dedicated disposable
evaluation host/container for untrusted adversarial code.

## Scoring and artifacts

- Math: final boxed answer, with the frozen equivalence rules.
- Multiple choice: extracted-letter exact match, not a semantic judge.
- SciBench: `scibench-numeric-units-v2`; balanced final box, bounded numeric
  arithmetic, compatible explicit units, 1% relative tolerance. It never
  evaluates model text with Python `eval` or unrestricted SymPy.
- Code: remove declared thought text, take the first Python fence, normalize
  full functions/body-only legacy outputs using signatures and prompt context,
  then run the **original tests**. No reference implementation body is injected
  and no intermediate code is selected by whether it passes tests.
- BBH choices: gold-independent explicit final-answer parsing. Object counting
  retains boxed/answer-marker normalized exact matching.
- BBH macro-average: equal weight for the three slice accuracies, calculated
  before rounding; not a pooled 1,750-question accuracy.

`results.json` records every seed's correct count and unrounded mean.
The public release's aggregate counts are in
[`model_release/gac_qwen35_4b/EVAL_RESULTS.json`](../../../model_release/gac_qwen35_4b/EVAL_RESULTS.json).
Dataset contents and raw predictions are local artifacts, not bundled public
data. Run synthetic parser/contract tests without GPUs or datasets:

```bash
python -m unittest discover -s eval/profiles/gac_release_20261008 -p 'test_*.py'
```
