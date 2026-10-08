"""Immutable data/decoding contract for the compact release snapshot."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

PROFILE_ID = 'gac-qwen35-4b-20261008'
SEEDS = (0, 1, 2)
SIZES = {
    'amc': 83, 'aime24': 30, 'aime25': 30, 'mmlu-pro': 1000,
    'gpqa': 198, 'scibench': 692, 'mbpp': 500, 'humaneval': 164,
    'logical_deduction': 750, 'object_counting': 250,
    'tracking_shuffled_objects': 750,
}
DATA_FINGERPRINT = 'b4e4095a62395764927efe2b7b04992b1018895f81532b6b909dfefc62ebee8e'
GENERATION = {
    'temperature': 0.6, 'top_p': 0.95, 'max_new_tokens': 8192,
    'max_model_len': 16384, 'tp_size': 4, 'dtype': 'bfloat16',
    'gpu_memory_utilization': 0.8, 'engine_seed': 0, 'n_samples': 1,
    'enable_prefix_caching': False, 'enable_chunked_prefill': True,
}


def read_jsonl(path: Path) -> list[dict]:
    # Physical lines, not str.splitlines(): prompts may contain U+2028/U+0085.
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def fingerprint(items: list[dict]) -> str:
    normalized = [
        {'benchmark': row['benchmark'], 'prompt': row['eval_item']['prompt'],
         'gold': row['eval_item'].get('gold'),
         'tests': row['eval_item'].get('test_list') or row['eval_item'].get('test')}
        for row in items
    ]
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_items(items: list[dict]) -> None:
    counts = collections.Counter(row['benchmark'] for row in items)
    if dict(counts) != SIZES:
        raise ValueError(f'Incomplete or different task slices: {dict(counts)}')
    keys = [(row['benchmark'], row['id']) for row in items]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate evaluation IDs')
    if [row['benchmark'] for row in items] != [
            key for key, n in SIZES.items() for _ in range(n)]:
        raise ValueError('Task order differs from the joint-batch release profile')
    actual = fingerprint(items)
    if actual != DATA_FINGERPRINT:
        raise ValueError(f'Dataset/order drift: expected {DATA_FINGERPRINT}, got {actual}')


def validate_predictions(rows: list[dict], items: list[dict]) -> None:
    expected = {(f'seed_{seed}', row['benchmark'], row['id'])
                for seed in SEEDS for row in items}
    actual = [(row['mode'], row['benchmark'], row['id']) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing, duplicate, or unknown predictions')
    if any(not isinstance(row.get('completion'), str) for row in rows):
        raise ValueError('Non-string completion')


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for task, n in SIZES.items():
        counts = []
        for seed in SEEDS:
            selected = [r for r in rows if r['benchmark'] == task
                        and r['mode'] == f'seed_{seed}']
            if len(selected) != n or len({r['id'] for r in selected}) != n:
                raise ValueError(f'Incomplete/duplicate score group: seed_{seed}/{task}')
            if any(type(r['correct']) is not bool for r in selected):
                raise ValueError('Scores must be Boolean, not coerced strings')
            counts.append(sum(r['correct'] for r in selected))
        percentages = [100 * count / n for count in counts]
        mean = sum(percentages) / len(SEEDS)
        result[task] = {'n_examples': n, 'correct_counts': counts,
                        'seed_percentages': percentages, 'mean_percentage': mean,
                        'display_percentage': round(mean, 1)}
    bbh = ('logical_deduction', 'object_counting', 'tracking_shuffled_objects')
    values = [sum(result[key]['seed_percentages'][s] for key in bbh) / 3
              for s in range(len(SEEDS))]
    mean = sum(values) / len(SEEDS)
    result['bbh_macro'] = {'n_slices': 3, 'n_examples': 1750,
                           'seed_percentages': values, 'mean_percentage': mean,
                           'display_percentage': round(mean, 1)}
    return {'profile': PROFILE_ID, 'seeds': list(SEEDS), 'metrics': result}
