"""Knowledge benchmark evaluation for the GAC harness.

Covers:
    - MMLU-Pro (multi-choice, up to 10 options, TIGER-Lab/MMLU-Pro)
    - GPQA-diamond (multi-choice, 4 options, Idavidrein/gpqa)
    - SciBench (open-ended numeric/symbolic, xw27/scibench)

The first two use a strict letter-match on \\boxed{X}. SciBench uses SymPy for
numeric/symbolic equivalence, with a string-normalization fallback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

from common import public_model_label, require_dataset_size
from generate_vllm import GenerationConfig, generate
from prompts import MCQ_SYSTEM, SCIENCE_SYSTEM, mcq_user_prompt, scibench_user_prompt


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
LETTER_RE = re.compile(r"\b([A-J])\b")
MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_PRO_MANIFEST = (
    Path(__file__).resolve().parent / "manifests"
    / f"mmlu_pro_test_{MMLU_PRO_REVISION}.json"
)


def extract_boxed(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_letter(text: str) -> str | None:
    """First try \\boxed{X}, then any of ``The answer is X`` / final capital letter."""
    boxed = extract_boxed(text)
    if boxed:
        letters = LETTER_RE.findall(boxed.upper())
        if letters:
            return letters[0]

    m = re.search(
        r"(?:the answer is|answer:|final answer)[\s:]*\(?([A-J])\)?",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    letters = LETTER_RE.findall(text[-200:].upper())
    return letters[-1] if letters else None


# ---------------------------- MMLU-Pro / GPQA (MCQ) ---------------------------


def _load_mmlu_pro(subset_size: int = 1000, seed: int = 42) -> list[dict]:
    """Load manifest-selected IDs at a pinned revision (seed-42 random sample).

    Selection is independent of the generation seed and dataset row order.
    The original selection algorithm was pseudorandom, not stratified.
    """
    manifest = json.loads(MMLU_PRO_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "dataset": MMLU_PRO_DATASET,
        "config": "default",
        "split": "test",
        "revision": MMLU_PRO_REVISION,
        "dataset_size": 12032,
        "id_field": "question_id",
        "sample_size": 1000,
        "seed": 42,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("MMLU-Pro: manifest metadata does not match the fixed protocol")
    if subset_size != manifest["sample_size"] or seed != manifest["seed"]:
        raise ValueError("MMLU-Pro: the release manifest fixes sample_size=1000 and seed=42")
    entries = manifest["entries"]
    require_dataset_size("MMLU-Pro manifest", len(entries), subset_size)
    ids = [entry["question_id"] for entry in entries]
    if len(set(ids)) != len(ids):
        raise RuntimeError("MMLU-Pro: duplicate question_id in manifest")

    ds = load_dataset(
        MMLU_PRO_DATASET, "default", split="test", revision=MMLU_PRO_REVISION
    )
    require_dataset_size("MMLU-Pro test split", len(ds), manifest["dataset_size"])
    rows_by_id = {}
    for row in ds:
        qid = row["question_id"]
        if qid in rows_by_id:
            raise RuntimeError(f"MMLU-Pro: duplicate question_id={qid} in dataset")
        rows_by_id[qid] = row

    items = []
    for entry in entries:
        qid = entry["question_id"]
        if qid not in rows_by_id:
            raise RuntimeError(f"MMLU-Pro: missing manifest question_id={qid}")
        row = rows_by_id[qid]
        raw_options = row["options"]
        if not isinstance(raw_options, (list, tuple)) or not 3 <= len(raw_options) <= 10:
            raise RuntimeError(
                f"MMLU-Pro question_id={qid}: expected 3-10 options, found "
                f"{len(raw_options) if isinstance(raw_options, (list, tuple)) else type(raw_options).__name__}"
            )
        if len(raw_options) != entry["n_options"] or row["category"] != entry["category"]:
            raise RuntimeError(f"MMLU-Pro question_id={qid}: dataset differs from manifest")
        answer_index = int(row["answer_index"])
        if not 0 <= answer_index < len(raw_options):
            raise RuntimeError(f"MMLU-Pro question_id={qid}: invalid answer_index={answer_index}")
        options = {chr(65 + j): opt for j, opt in enumerate(raw_options)}
        # answer_index in MMLU-Pro is 0-indexed integer
        gold_letter = chr(65 + answer_index)
        items.append(
            {
                "id": f"mmlu-pro_{qid}",
                "question_id": qid,
                "category": row["category"],
                "prompt": mcq_user_prompt(row["question"], options),
                "gold": gold_letter,
                "raw_question": row["question"],
                "options": options,
            }
        )
    return items


def _load_gpqa_diamond(csv_path: str | None = None) -> tuple[list[dict], str]:
    """Load gated GPQA-Diamond or a user-provided, lawfully obtained CSV.

    A clean checkout does not include GPQA data. Authenticate with Hugging
    Face after obtaining dataset access, or provide the complete Diamond
    split through ``--gpqa_csv`` / ``GAC_GPQA_CSV``.
    """
    csv_path = csv_path or os.environ.get("GAC_GPQA_CSV")
    if csv_path:
        path = Path(csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GPQA CSV not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        require_dataset_size("GPQA-Diamond", len(rows), 198)
        required = {
            "Question",
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
        }
        missing = sorted(required - set(rows[0] if rows else []))
        if missing:
            raise ValueError(f"GPQA CSV missing columns: {missing}")
        ds = rows
        source = "csv:provided_file:sha256=" + hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Unable to load GPQA-Diamond. It is gated on Hugging Face: "
                "obtain access at https://huggingface.co/datasets/Idavidrein/gpqa "
                "and authenticate with `hf auth login`, or supply a lawfully "
                "obtained 198-row CSV via --gpqa_csv / GAC_GPQA_CSV. "
                "No GPQA data is bundled in this repository. "
                "Check the underlying error for connectivity or cache failures."
            ) from exc
        require_dataset_size("GPQA-Diamond", len(ds), 198)
        source = "hf:Idavidrein/gpqa:gpqa_diamond:train"

    items = []
    for i, row in enumerate(ds):
        # GPQA has one correct and three incorrect answers as separate columns.
        opts = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        rng = random.Random(i)
        perm = list(range(4))
        rng.shuffle(perm)
        shuffled = [opts[p] for p in perm]
        correct_idx = perm.index(0)
        letter = chr(65 + correct_idx)
        options_dict = {chr(65 + j): shuffled[j] for j in range(4)}
        items.append(
            {
                "id": f"gpqa_{i}",
                "prompt": mcq_user_prompt(row["Question"], options_dict),
                "gold": letter,
                "raw_question": row["Question"],
                "options": options_dict,
            }
        )
    return items, source


def _score_mcq(completion: str, gold: str) -> bool:
    pred = extract_letter(completion)
    return pred is not None and pred == gold.upper()


# --------------------------------- SciBench -----------------------------------


def _load_scibench() -> list[dict]:
    ds = load_dataset("xw27/scibench", split="train")
    items = []
    for i, row in enumerate(ds):
        problem = row.get("problem_text") or row.get("problem") or row.get("question")
        gold = row.get("answer_number")
        if gold is None:
            gold = row.get("answer")
        if gold is None:
            gold = row.get("gold_answer")
        unit = row.get("unit")
        if problem is None or gold is None:
            continue
        items.append(
            {
                "id": f"scibench_{i}",
                "prompt": scibench_user_prompt(str(problem), unit),
                "gold": str(gold),
                "raw_problem": str(problem),
                "unit": unit,
            }
        )
    require_dataset_size("SciBench", len(items), 692)
    return items


def _score_scibench(completion: str, gold: str) -> bool:
    pred = extract_boxed(completion)
    if pred is None:
        return False

    # Try SymPy equivalence.
    try:
        import sympy as sp
        g = sp.sympify(str(gold).replace(",", ""))
        p = sp.sympify(pred.replace(",", ""))
        if sp.simplify(g - p) == 0:
            return True
    except Exception:
        pass

    # Numeric fallback: compare as floats with 1% tolerance.
    try:
        g = float(str(gold).replace(",", ""))
        p = float(pred.replace(",", ""))
        return abs(g - p) / max(1.0, abs(g)) < 0.01
    except Exception:
        pass

    # String fallback: normalize whitespace + trailing punctuation.
    return pred.strip().rstrip(".").replace(" ", "") == str(gold).strip().rstrip(".").replace(" ", "")


# --------------------------------- Runner -------------------------------------


LOADERS = {
    "mmlu-pro": (_load_mmlu_pro, _score_mcq, MCQ_SYSTEM),
    "gpqa": (_load_gpqa_diamond, _score_mcq, MCQ_SYSTEM),
    "scibench": (_load_scibench, _score_scibench, SCIENCE_SYSTEM),
}


def run_benchmark(
    name: str,
    model_path: str,
    output_dir: Path,
    cfg: GenerationConfig,
    gpqa_csv: str | None = None,
) -> dict:
    loader, scorer, system_prompt = LOADERS[name]
    source = None
    if name == "gpqa":
        items, source = _load_gpqa_diamond(gpqa_csv)
    else:
        items = loader()
    completions = generate(
        [it["prompt"] for it in items], cfg, system_prompt=system_prompt
    )

    records = []
    correct = 0
    for it, cs in zip(items, completions):
        best = any(scorer(c, it["gold"]) for c in cs)
        records.append(
            {
                "id": it["id"],
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
        "data_source": source
        or {
            "mmlu-pro": f"hf:{MMLU_PRO_DATASET}:test@{MMLU_PRO_REVISION}",
            "scibench": "hf:xw27/scibench:train",
        }[name],
    }
    if name == "mmlu-pro":
        summary["dataset_revision"] = MMLU_PRO_REVISION
        summary["subset_manifest"] = MMLU_PRO_MANIFEST.name
        summary["subset_manifest_sha256"] = hashlib.sha256(
            MMLU_PRO_MANIFEST.read_bytes()
        ).hexdigest()
        summary["subset_seed"] = 42
        summary["id_field"] = "question_id"
    with open(per_bench_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[knowledge_bench] {name}: acc={acc*100:.1f}% ({correct}/{len(records)}) → {per_bench_dir}"
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path")
    p.add_argument("--output_dir")
    p.add_argument(
        "--check_data", action="store_true",
        help="Validate selected datasets on CPU, without loading a model or generating answers.",
    )
    p.add_argument(
        "--benchmarks", nargs="+", default=list(LOADERS.keys()),
        choices=list(LOADERS.keys()),
    )
    p.add_argument("--tp_size", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpqa_csv",
        default=None,
        help="User-provided GPQA-Diamond CSV (requires lawful access); also read from GAC_GPQA_CSV.",
    )
    args = p.parse_args()

    if args.check_data:
        failures = []
        for name in args.benchmarks:
            try:
                if name == "gpqa":
                    items, source = _load_gpqa_diamond(args.gpqa_csv)
                else:
                    items = LOADERS[name][0]()
                    source = (
                        f"hf:{MMLU_PRO_DATASET}:test@{MMLU_PRO_REVISION}"
                        if name == "mmlu-pro" else "hf:xw27/scibench:train"
                    )
                report = {
                    "benchmark": name, "status": "validated",
                    "n_examples": len(items), "data_source": source,
                }
                if name == "mmlu-pro":
                    report["subset_manifest_sha256"] = hashlib.sha256(
                        MMLU_PRO_MANIFEST.read_bytes()
                    ).hexdigest()
                print(json.dumps(report))
            except Exception as exc:
                failures.append(name)
                print(f"[check_data] {name}: {exc}", file=sys.stderr)
        if failures:
            raise SystemExit(1)
        return

    if not args.model_path or not args.output_dir:
        p.error("--model_path and --output_dir are required unless --check_data is used")

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
    failures = {}
    for name in args.benchmarks:
        try:
            all_summaries[name] = run_benchmark(
                name, args.model_path, out, cfg, gpqa_csv=args.gpqa_csv
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures[name] = message
            all_summaries[name] = {
                "benchmark": name,
                "model": public_model_label(args.model_path),
                "seed": args.seed,
                "status": "failed",
                "error": message,
            }
            print(f"[knowledge_bench] {name}: FAILED — {message}", file=sys.stderr)

    with open(out / "knowledge_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    if failures:
        print(
            f"[knowledge_bench] completed with failures: {', '.join(failures)}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
