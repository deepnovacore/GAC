# GAC Release Notes

## 2026-09-11 — Fixed dataset protocol and release corrections

- Pin MMLU-Pro to revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`
  and publish all 1,000 selected question IDs. Preserve the seed-42 random
  selection algorithm, describe it accurately, and validate actual option
  counts instead of requiring every question to have 10 options.
- Record the manifest digest in evaluation summaries and support CPU-only
  dataset preflight through `knowledge_bench.py --check_data`.
- Document gated GPQA access and user-provided CSV support; remove the
  incorrect claim that a public CSV is bundled. CSV summaries include a digest.
- Correct the author-reported compact release values: AMC 67.5%, AIME25 20.0%,
  HumanEval 81.1%. These are publication corrections, not newly generated
  benchmark scores. Refresh the Qwen3.5-4B model-card introduction and retain
  the evaluation protocol and single-snapshot interpretation.

## v0.1.1 (2026-09) — Public evaluation and compact-model release

This release turns the research evaluator into a portable, auditable public
artifact and adds release metadata for the GAC-Qwen3.5-4B compact model.

### Highlights

- Public evaluator covering math, knowledge, science, code, and BBH logic task
  slices.
- Dataset-source recording and expected-size validation to prevent scores from
  silently using partial mirrors.
- Code generation and code execution are separate stages; generated Python is
  executed only through an isolated user/mount/network/PID namespace wrapper.
- Evaluation summaries store a short model label rather than a local absolute
  path.
- Aggregate evaluation metadata and model-card templates are ready for the
  Hugging Face and ModelScope model repositories.

### Scope and reproducibility

The public repository contains evaluator source, scoring logic, documentation,
and tests. It does not contain benchmark prompts, gold answers, raw model
generations, execution traces, optimizer states, or private infrastructure
configuration. Users should follow each benchmark's license and record the
dataset revision, decoding configuration, seed, and scorer version when
reporting results.

## v0.1.0 (2026-08) — Initial release

- GAC controller and hybrid-loss modules.
- Reference configuration and unit tests.
- Initial checkpoint evaluation harness.
- Academic project page and camera-ready paper materials.
