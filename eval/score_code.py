"""Score saved MBPP/HumanEval generations in a separately isolated process.

The upstream BigCode evaluator executes model-generated Python and explicitly
warns that its executor is not a security sandbox.  This script therefore
refuses to run unless ``sandbox_code_eval.sh`` has set
``GAC_CODE_EVAL_SANDBOX=1`` and datasets are in offline mode.  It consumes the
``generations.json`` files emitted by :mod:`code_bench` and writes a compact
summary plus per-candidate execution results.

The isolation wrapper is intentionally separate from generation: no generated
code is executed by a vLLM/GPU job or on a login node.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset


def _load_generations(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8") as f:
        generations = json.load(f)
    if not isinstance(generations, list):
        raise ValueError(f"Expected a list in {path}")
    normalized: list[list[str]] = []
    for i, candidates in enumerate(generations):
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Task {i} in {path} has no candidate list")
        if not all(isinstance(candidate, str) for candidate in candidates):
            raise ValueError(f"Task {i} in {path} contains a non-string candidate")
        normalized.append(candidates)
    return normalized


def _load_references(name: str, mbpp_config: str) -> tuple[list[str], str]:
    if name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        references = [
            "\n" + row["test"] + "\n" + f"check({row['entry_point']})"
            for row in ds
        ]
        return references, "openai/openai_humaneval:test"

    if name == "mbpp":
        ds = load_dataset(
            "google-research-datasets/mbpp", mbpp_config, split="test"
        )
        references = ["\n".join(row["test_list"]) for row in ds]
        return references, f"google-research-datasets/mbpp:{mbpp_config}:test"

    raise ValueError(f"Unsupported code benchmark: {name}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def score_one(
    name: str,
    output_dir: Path,
    bigcode_root: Path,
    mbpp_config: str,
    num_workers: int,
    timeout: float,
) -> dict:
    if os.environ.get("GAC_CODE_EVAL_SANDBOX") != "1":
        raise RuntimeError(
            "Refusing to execute generated code outside sandbox_code_eval.sh "
            "(GAC_CODE_EVAL_SANDBOX=1 is required)."
        )
    if os.environ.get("HF_DATASETS_OFFLINE") not in {"1", "true", "TRUE"}:
        raise RuntimeError("HF_DATASETS_OFFLINE=1 is required for code scoring")

    sys.path.insert(0, str(bigcode_root))
    # Import only after the guard: this is the first point at which the
    # upstream execution metric is made available.
    from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

    gen_path = output_dir / name / "generations.json"
    if not gen_path.is_file():
        raise FileNotFoundError(f"Missing generations file: {gen_path}")
    generations = _load_generations(gen_path)
    references, data_source = _load_references(name, mbpp_config)
    if len(generations) != len(references):
        raise ValueError(
            f"{name}: {len(generations)} generations but "
            f"{len(references)} references ({data_source})"
        )

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    pass_at_k, details = compute_code_eval(
        predictions=generations,
        references=references,
        k=[1],
        num_workers=num_workers,
        timeout=timeout,
    )

    per_dir = output_dir / name
    granular = []
    for task_id in sorted(details):
        results = details[task_id]
        granular.append(
            {
                "task_index": int(task_id),
                "candidates": [
                    {
                        "completion_id": int(completion_id),
                        "passed": bool(result["passed"]),
                        "result": str(result["result"]),
                    }
                    for completion_id, result in sorted(results)
                ],
            }
        )
    _write_json(per_dir / "execution_results.json", granular)

    summary = {
        "benchmark": name,
        "n_total": len(generations),
        "n_samples": len(generations[0]) if generations else 0,
        "pass@1": float(pass_at_k["pass@1"]),
        "n_tasks_with_a_pass": sum(
            any(candidate["passed"] for candidate in task["candidates"])
            for task in granular
        ),
        "data_source": data_source,
        "scorer": "bigcode-evaluation-harness",
        "scorer_commit": "8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd",
        "timeout_seconds": timeout,
        "num_workers": num_workers,
        "status": "scored_in_isolated_user_mount_net_pid_namespace",
    }
    _write_json(per_dir / "summary.json", summary)
    print(
        f"[score_code] {name}: pass@1={summary['pass@1'] * 100:.2f}% "
        f"({summary['n_tasks_with_a_pass']}/{summary['n_total']})"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bigcode_root", required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["mbpp", "humaneval"],
        default=["mbpp", "humaneval"],
    )
    parser.add_argument(
        "--mbpp_config",
        choices=["full", "sanitized"],
        default="full",
        help="MBPP test variant; full=500 (paper-aligned), sanitized=257.",
    )
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    bigcode_root = Path(args.bigcode_root).resolve()
    summaries = {}
    for name in args.benchmarks:
        summaries[name] = score_one(
            name=name,
            output_dir=output_dir,
            bigcode_root=bigcode_root,
            mbpp_config=args.mbpp_config,
            num_workers=args.num_workers,
            timeout=args.timeout,
        )
    _write_json(output_dir / "code_summary.json", summaries)


if __name__ == "__main__":
    main()
