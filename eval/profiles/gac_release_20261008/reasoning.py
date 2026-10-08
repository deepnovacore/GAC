"""Frozen answer rules; no judge or reference-dependent answer repair."""
from __future__ import annotations

import re

from bbh_choice_protocol import extract_choice, options_from_prompt
from science_answer import score_science

BOXED_RE = re.compile(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}')
LETTER_RE = re.compile(r'\b([A-J])\b')
OBJECT_ANSWER_RE = re.compile(r'(?:the answer is)\s*[:\-]?\s*(.+?)[\s.]*$',
                             re.IGNORECASE | re.MULTILINE)
OBJECT_BOXED_RE = re.compile(r'\\boxed\s*\{([^{}]+)\}', re.IGNORECASE)


def extract_boxed(text):
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def score_math(text, gold):
    from math_verify import parse, verify
    prediction = extract_boxed(text)
    if prediction is None:
        return False
    try:
        return bool(verify(parse(str(gold)), parse(f'\\boxed{{{prediction}}}')))
    except Exception:
        return prediction.strip().rstrip('.').strip() == str(gold).strip().rstrip('.').strip()


def extract_letter(text):
    boxed = extract_boxed(text)
    if boxed:
        letters = LETTER_RE.findall(boxed.upper())
        if letters:
            return letters[0]
    match = re.search(r'(?:the answer is|answer:|final answer)[\s:]*\(?([A-J])\)?',
                      text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    letters = LETTER_RE.findall(text[-200:].upper())
    return letters[-1] if letters else None


def object_answer(text):
    candidates = [(m.start(), m[1]) for m in OBJECT_ANSWER_RE.finditer(text)]
    candidates.extend((m.start(), m[1]) for m in OBJECT_BOXED_RE.finditer(text))
    if candidates:
        answer = max(candidates, key=lambda x: x[0])[1]
        answer = re.sub(r'\\text\s*\{([^{}]*)\}', r'\1', answer)
        return answer.strip().strip('().').strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def normalized(value):
    return re.sub(r'\s+', ' ', value.strip().lower()).strip('().').strip()


def score(task, completion, item):
    gold = item['gold']
    if task in ('amc', 'aime24', 'aime25'):
        return score_math(completion, gold)
    if task in ('mmlu-pro', 'gpqa'):
        prediction = extract_letter(completion)
        return prediction is not None and prediction == gold.upper()
    if task == 'scibench':
        return score_science(completion, gold, item.get('unit')).correct
    if task in ('logical_deduction', 'tracking_shuffled_objects'):
        prediction = extract_choice(completion, options_from_prompt(item['prompt']))
        return prediction is not None and prediction == gold.strip('(). ')
    if task == 'object_counting':
        prediction = object_answer(completion)
        return prediction is not None and normalized(prediction) == normalized(gold)
    raise ValueError(f'Unsupported reasoning task: {task}')
