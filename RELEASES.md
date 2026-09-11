# GAC Release Notes

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
