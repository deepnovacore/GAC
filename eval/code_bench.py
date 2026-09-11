"""Code generation benchmark for the GAC harness (MBPP + HumanEval).

This command only generates and stores completions.  Executing
model-generated Python is deliberately a separate step (``score_code.py``),
which must be launched inside the repository's Slurm sandbox wrapper.  The
upstream BigCode harness explicitly warns that its executor is not a security
sandbox, so scoring must never happen implicitly in this generator or on a
login node.

Usage:
    python code_bench.py \\
        --model_path <ckpt> --benchmarks mbpp humaneval \\
        --output_dir ./results --tp_size 4 --n_samples 1 --generation_only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from common import public_model_label, require_dataset_size
from generate_vllm import GenerationConfig, generate
from prompts import CODE_SYSTEM, humaneval_user_prompt, mbpp_user_prompt


PY_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str:
    """Extract a python code block from a completion. Fall back to raw text."""
    m = PY_BLOCK_RE.search(completion)
    if m:
        return m.group(1).strip()
    return completion.strip()


def _load_humaneval() -> list[dict]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    require_dataset_size("HumanEval", len(ds), 164)
    return [
        {
            "task_id": row["task_id"],
            "prompt": humaneval_user_prompt(row["prompt"]),
            "raw_prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }
        for row in ds
    ]


def _load_mbpp(config: str = "full") -> list[dict]:
    """Load the standard 500-question MBPP test split by default.

    ``sanitized`` is kept as an explicit option because it is a commonly used
    257-question variant, but it is not the 500-question split reported by the
    paper's MBPP table.
    """
    ds = load_dataset("google-research-datasets/mbpp", config, split="test")
    require_dataset_size(
        f"MBPP/{config}", len(ds), 500 if config == "full" else 257
    )
    text_key = "text" if config == "full" else "prompt"
    return [
        {
            "task_id": f"mbpp_{row['task_id']}",
            "prompt": mbpp_user_prompt(row[text_key], row["test_list"]),
            "raw_text": row[text_key],
            "test_list": row["test_list"],
            "code": row.get("code"),
        }
        for row in ds
    ]


LOADERS = {"humaneval": _load_humaneval, "mbpp": _load_mbpp}


def _write_bigcode_generations(
    items: list[dict],
    completions: list[list[str]],
    path: Path,
) -> None:
    """Write generations in the shape bigcode-evaluation-harness expects
    when loaded via ``--load_generations_path``: a list-of-lists JSON."""
    all_gens = [[extract_code(c) for c in cs] for cs in completions]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_gens, f)


def run_benchmark(
    name: str,
    model_path: str,
    output_dir: Path,
    cfg: GenerationConfig,
    mbpp_config: str = "full",
) -> dict:
    items = _load_mbpp(mbpp_config) if name == "mbpp" else LOADERS[name]()
    completions = generate(
        [it["prompt"] for it in items], cfg, system_prompt=CODE_SYSTEM
    )

    per_bench_dir = output_dir / name
    per_bench_dir.mkdir(parents=True, exist_ok=True)

    # Save raw predictions.
    with open(per_bench_dir / "predictions.jsonl", "w") as f:
        for it, cs in zip(items, completions):
            f.write(
                json.dumps(
                    {
                        "task_id": it["task_id"],
                        "completions": cs,
                        "extracted_code": [extract_code(c) for c in cs],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Emit bigcode-compatible generations file for the separate scorer.
    gen_path = per_bench_dir / "generations.json"
    _write_bigcode_generations(items, completions, gen_path)

    summary = {
        "benchmark": name,
        "model": public_model_label(model_path),
        "n_total": len(items),
        "seed": cfg.seed,
        "n_samples": cfg.n_samples,
        "status": "generated",
        "pass@1": None,
        "data_source": (
            f"google-research-datasets/mbpp:{mbpp_config}:test"
            if name == "mbpp"
            else "openai/openai_humaneval:test"
        ),
        # Keep summaries portable and safe to publish; the raw file itself is
        # intentionally local because it contains benchmark prompts and code.
        "generations_file": "generations.json",
        "scoring": (
            "Run score_code.py through sandbox_code_eval.sh; generated code is "
            "not executed by this command."
        ),
    }

    with open(per_bench_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[code_bench] {name}: summary → {per_bench_dir / 'summary.json'}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--benchmarks", nargs="+", default=list(LOADERS.keys()),
        choices=list(LOADERS.keys()),
    )
    p.add_argument("--tp_size", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)  # code: lower T
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mbpp_config",
        choices=["full", "sanitized"],
        default="full",
        help="MBPP test variant; full=500 (paper-aligned), sanitized=257.",
    )
    p.add_argument(
        "--generation_only",
        action="store_true",
        help="Accepted for explicitness; code_bench always stops before execution.",
    )
    args = p.parse_args()

    cfg = GenerationConfig(
        model_path=args.model_path,
        tp_size=args.tp_size,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for name in args.benchmarks:
        all_summaries[name] = run_benchmark(
            name, args.model_path, out, cfg, mbpp_config=args.mbpp_config
        )

    with open(out / "code_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
