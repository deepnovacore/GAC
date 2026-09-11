"""Prompt templates for the GAC eval harness.

Templates match the paper's evaluation setup (Sec. 4.1). We keep them minimal
and match the LUFFY / DeepSeek-R1 / Qwen2.5-Math convention of asking the model
to produce its final answer inside ``\\boxed{...}`` so parsers can extract it.
"""

from __future__ import annotations

# System prompt for reasoning-style math benchmarks (AMC, AIME).
MATH_SYSTEM = (
    "You are a helpful assistant. Solve the following math problem step by step. "
    "Put your final answer inside \\boxed{}."
)

# System prompt for multi-choice knowledge tasks (MMLU-Pro, GPQA).
MCQ_SYSTEM = (
    "You are a helpful assistant. Read the question and options carefully, "
    "think step by step, and put the letter of your final answer inside "
    "\\boxed{}. For example: \\boxed{A}."
)

# System prompt for open-ended scientific problems (SciBench).
SCIENCE_SYSTEM = (
    "You are a helpful assistant. Solve the following scientific problem "
    "step by step. Provide your final numeric or symbolic answer inside "
    "\\boxed{}."
)

# System prompt for BBH-Logic subsets. BBH uses a specific "The answer is X" style.
BBH_LOGIC_SYSTEM = (
    "You are a helpful assistant. Read the problem, reason step by step, "
    "and end with 'The answer is X' where X is your final choice."
)

# System prompt for code generation (MBPP / HumanEval).
CODE_SYSTEM = (
    "You are an expert Python programmer. Complete the following function. "
    "Only output the function body (or the full function if requested). "
    "Do not include example usage or tests."
)


def math_user_prompt(problem: str) -> str:
    return f"Problem: {problem}\n\nSolution:"


def mcq_user_prompt(question: str, options: dict[str, str]) -> str:
    """options: {'A': 'text', 'B': 'text', ...}"""
    lines = [f"Question: {question}", "", "Options:"]
    for letter, text in options.items():
        lines.append(f"({letter}) {text}")
    lines.append("")
    lines.append("Which is correct?")
    return "\n".join(lines)


def scibench_user_prompt(problem: str, unit: str | None = None) -> str:
    tail = f"\n\nReport your answer in units of {unit}." if unit else ""
    return f"Problem: {problem}{tail}\n\nSolution:"


def bbh_user_prompt(problem: str) -> str:
    return problem  # BBH inputs already contain the format instruction.


def humaneval_user_prompt(prompt: str) -> str:
    return prompt  # canonical HumanEval prompts are self-contained.


def mbpp_user_prompt(text: str, test_list: list[str]) -> str:
    tests = "\n".join(test_list)
    return (
        f"{text}\n\nYour code should satisfy these tests:\n{tests}\n\n"
        "```python\n"
    )
