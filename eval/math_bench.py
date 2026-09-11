"""Math benchmark evaluation for the GAC harness.

Supports AMC / AIME24 / AIME25 (paper Table 1). Uses ``math-verify`` for
robust equivalence checking that ignores formatting differences
(e.g. ``\\frac{1}{2}`` vs ``0.5``, ``\\sqrt{2}`` vs ``2**0.5``).

Usage:
    python math_bench.py \\
        --model_path <ckpt> --benchmarks amc aime24 aime25 \\
        --output_dir ./results --tp_size 4 --seed 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from common import public_model_label, require_dataset_size
from generate_vllm import GenerationConfig, generate
from prompts import MATH_SYSTEM, math_user_prompt


try:
    from math_verify import parse, verify  # type: ignore
except ImportError:
    parse = None  # type: ignore
    verify = None  # type: ignore


# HuggingFace dataset IDs. Splits vary; we normalize downstream.
DATASETS = {
    "amc": ("AI-MO/aimo-validation-amc", "train", "problem", "answer"),
    "aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
    "aime25": ("math-ai/aime25", "test", "problem", "answer"),
}

EXPECTED_SIZES = {"amc": 83, "aime24": 30, "aime25": 30}


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text: str) -> str | None:
    """Return the last \\boxed{...} content in ``text`` (LUFFY convention)."""
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def score_one(completion: str, gold: str) -> bool:
    """Score a single completion against gold. Uses ``math-verify`` if available,
    otherwise falls back to string comparison of ``\\boxed{}`` content.
    """
    pred = extract_boxed(completion)
    if pred is None:
        return False

    if parse is not None and verify is not None:
        try:
            g = parse(str(gold))
            p = parse(f"\\boxed{{{pred}}}")
            return bool(verify(g, p))
        except Exception:
            pass  # fall through to string match

    # Cheap fallback: normalize whitespace + strip trailing punctuation.
    return pred.strip().rstrip(".").strip() == str(gold).strip().rstrip(".").strip()


def load_benchmark(name: str) -> list[dict]:
    if name not in DATASETS:
        raise ValueError(f"Unknown math benchmark: {name!r}")
    hf_id, split, prob_key, ans_key = DATASETS[name]
    ds = load_dataset(hf_id, split=split)
    require_dataset_size(name, len(ds), EXPECTED_SIZES[name])
    out = []
    for i, row in enumerate(ds):
        out.append(
            {
                "id": f"{name}_{i}",
                "prompt": math_user_prompt(row[prob_key]),
                "gold": str(row[ans_key]),
                "raw_problem": row[prob_key],
            }
        )
    return out


def run_benchmark(
    name: str, model_path: str, output_dir: Path, cfg: GenerationConfig
) -> dict:
    items = load_benchmark(name)
    completions = generate(
        [it["prompt"] for it in items], cfg, system_prompt=MATH_SYSTEM
    )

    records = []
    correct = 0
    for it, cs in zip(items, completions):
        best_correct = any(score_one(c, it["gold"]) for c in cs)
        records.append(
            {
                "id": it["id"],
                "problem": it["raw_problem"],
                "gold": it["gold"],
                "completions": cs,
                "correct": best_correct,
            }
        )
        correct += int(best_correct)

    acc = correct / max(1, len(records))
    per_bench_dir = output_dir / name
    per_bench_dir.mkdir(parents=True, exist_ok=True)

    with open(per_bench_dir / "predictions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "benchmark": name,
        "model": public_model_label(model_path),
        "n_total": len(records),
        "n_correct": correct,
        "accuracy": acc,
        "seed": cfg.seed,
        "temperature": cfg.temperature,
        "n_samples": cfg.n_samples,
        "data_source": f"{DATASETS[name][0]}:{DATASETS[name][1]}",
    }
    with open(per_bench_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[math_bench] {name}: acc={acc*100:.1f}% ({correct}/{len(records)}) → {per_bench_dir}"
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--benchmarks", nargs="+", default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    p.add_argument("--tp_size", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
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
        all_summaries[name] = run_benchmark(name, args.model_path, out, cfg)

    with open(out / "math_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[math_bench] done. summary → {out / 'math_summary.json'}")


if __name__ == "__main__":
    main()
