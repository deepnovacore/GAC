"""Unified vLLM-based batch generator for the GAC evaluation harness.

All benchmark scripts (``math_bench.py``, ``knowledge_bench.py``, etc.) delegate
generation to :func:`generate` below so that sampling settings, chat templating,
and I/O are consistent across the reported task slices.

Sampling defaults (temperature=0.6, top_p=0.95, max_new_tokens=8192) match the
setup described in Sec. 4.1 of the GAC paper and are aligned with the LUFFY /
Qwen2.5-Math evaluation conventions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise SystemExit(
        "vllm is required. Install with:  pip install -r eval/requirements.txt"
    ) from e

from transformers import AutoTokenizer


@dataclasses.dataclass
class GenerationConfig:
    model_path: str
    tp_size: int = 4
    dtype: str = "bfloat16"
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.9
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 8192
    n_samples: int = 1
    seed: int = 0
    chat_template: str | None = None
    apply_chat_template: bool = True
    trust_remote_code: bool = True


def _render_prompts(
    tokenizer,
    prompts: Sequence[str],
    system_prompt: str | None,
    apply_chat_template: bool,
) -> list[str]:
    if not apply_chat_template:
        return list(prompts)

    rendered = []
    for user_msg in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_msg})
        rendered.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return rendered


def generate(
    prompts: Sequence[str],
    cfg: GenerationConfig,
    system_prompt: str | None = None,
) -> list[list[str]]:
    """Generate ``cfg.n_samples`` completions for each prompt.

    Returns a list of length ``len(prompts)``, where each element is a list of
    strings (``n_samples`` completions for that prompt).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=cfg.trust_remote_code
    )
    rendered = _render_prompts(
        tokenizer, prompts, system_prompt, cfg.apply_chat_template
    )

    llm = LLM(
        model=cfg.model_path,
        tensor_parallel_size=cfg.tp_size,
        dtype=cfg.dtype,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=cfg.trust_remote_code,
        seed=cfg.seed,
    )
    sampling = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        n=cfg.n_samples,
        seed=cfg.seed,
    )
    outputs = llm.generate(rendered, sampling)
    return [[o.text for o in out.outputs] for out in outputs]


def load_prompts_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_completions_jsonl(
    records: Iterable[dict], path: str | Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--prompts_jsonl", required=True, help="jsonl with {'id':..., 'prompt':...} per line")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--system_prompt", default=None)
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
    records = load_prompts_jsonl(args.prompts_jsonl)
    prompts = [r["prompt"] for r in records]
    completions = generate(prompts, cfg, system_prompt=args.system_prompt)

    out = []
    for r, cs in zip(records, completions):
        out.append({**r, "completions": cs})
    write_completions_jsonl(out, args.output_jsonl)
    print(f"[generate_vllm] wrote {len(out)} records → {args.output_jsonl}")


if __name__ == "__main__":
    cli()
