"""BBH logical-reasoning subsets for the GAC harness (paper Table 4).

Uses three subsets from Big-Bench-Hard:
    - logical_deduction (three_objects / five_objects / seven_objects merged)
    - object_counting
    - tracking_shuffled_objects (three_objects / five_objects / seven_objects merged)

Scoring: after asking the model to end with 'The answer is X', we extract the
final answer via a regex and do case-insensitive exact match against gold.
This matches the canonical BBH evaluation protocol.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from common import public_model_label
from generate_vllm import GenerationConfig, generate
from prompts import BBH_LOGIC_SYSTEM, bbh_user_prompt


# BBH task IDs per family (`lukaemon/bbh`).
FAMILIES = {
    "logical_deduction": [
        "logical_deduction_three_objects",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
    ],
    "object_counting": ["object_counting"],
    "tracking_shuffled_objects": [
        "tracking_shuffled_objects_three_objects",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
    ],
}


ANSWER_RE = re.compile(r"(?:the answer is)\s*[:\-]?\s*(.+?)[\s.]*$", re.IGNORECASE | re.MULTILINE)


def extract_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip().strip("().").strip()
    # Fallback: last non-empty line.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else None


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip("().").strip()


def score_one(completion: str, gold: str) -> bool:
    pred = extract_answer(completion)
    if pred is None:
        return False
    return normalize(pred) == normalize(gold)


def load_family(name: str) -> list[dict]:
    items = []
    failures = []
    for task in FAMILIES[name]:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
        except Exception as e:  # pragma: no cover
            failures.append(f"{task}: {type(e).__name__}: {e}")
            continue
        if len(ds) != 250:
            failures.append(f"{task}: expected 250 rows, found {len(ds)}")
            continue
        for i, row in enumerate(ds):
            items.append(
                {
                    "id": f"{task}_{i}",
                    "task": task,
                    "prompt": bbh_user_prompt(row["input"]),
                    "gold": str(row["target"]),
                }
            )
    if failures:
        raise RuntimeError(
            "BBH dataset validation failed; refusing a misleading partial score:\n"
            + "\n".join(failures)
        )
    return items


def run_family(
    name: str, model_path: str, output_dir: Path, cfg: GenerationConfig
) -> dict:
    items = load_family(name)
    completions = generate(
        [it["prompt"] for it in items], cfg, system_prompt=BBH_LOGIC_SYSTEM
    )

    records = []
    correct = 0
    for it, cs in zip(items, completions):
        best = any(score_one(c, it["gold"]) for c in cs)
        records.append(
            {
                "id": it["id"],
                "task": it["task"],
                "gold": it["gold"],
                "completions": cs,
                "correct": best,
            }
        )
        correct += int(best)

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
        "data_source": ",".join(
            f"lukaemon/bbh:{task}:test" for task in FAMILIES[name]
        ),
    }
    with open(per_bench_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[bbh_logic] {name}: acc={acc*100:.1f}% ({correct}/{len(records)}) → {per_bench_dir}"
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--benchmarks", nargs="+", default=list(FAMILIES.keys()),
        choices=list(FAMILIES.keys()),
    )
    p.add_argument("--tp_size", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=4096)
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
        all_summaries[name] = run_family(name, args.model_path, out, cfg)

    # Aggregate BBH-Logic average (paper Table 4 reports this).
    accs = [s["accuracy"] for s in all_summaries.values() if s.get("accuracy") is not None]
    if accs:
        all_summaries["_bbh_logic_avg"] = {"accuracy": sum(accs) / len(accs), "n_families": len(accs)}

    with open(out / "bbh_logic_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
