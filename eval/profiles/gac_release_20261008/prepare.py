"""CPU-only dataset preparation; no GPQA data is distributed with this repo."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import frozen_prompts as prompts
from protocol import DATA_FINGERPRINT, PROFILE_ID, validate_items, write_json

EVAL_ROOT = Path(__file__).resolve().parents[2]


def load_items(gpqa_csv: str | None = None) -> list[dict]:
    sys.path.insert(0, str(EVAL_ROOT))
    import math_bench
    import knowledge_bench
    import code_bench
    import bbh_logic

    # Freeze loader prompt hooks; leave the general harness defaults untouched.
    math_bench.math_user_prompt = prompts.math_user_prompt
    knowledge_bench.mcq_user_prompt = prompts.mcq_user_prompt
    knowledge_bench.scibench_user_prompt = prompts.scibench_user_prompt
    code_bench.mbpp_user_prompt = prompts.mbpp_user_prompt
    code_bench.humaneval_user_prompt = prompts.humaneval_user_prompt
    bbh_logic.bbh_user_prompt = prompts.bbh_user_prompt
    items = []

    def append(task, source, system):
        for item in source:
            items.append({'benchmark': task, 'id': item.get('id', item.get('task_id')),
                          'eval_item': item,
                          'messages': [{'role': 'system', 'content': system},
                                       {'role': 'user', 'content': item['prompt']}]})

    for task in ('amc', 'aime24', 'aime25'):
        append(task, math_bench.load_benchmark(task), prompts.MATH_SYSTEM)
    append('mmlu-pro', knowledge_bench._load_mmlu_pro(), prompts.MCQ_SYSTEM)
    gpqa, _ = knowledge_bench._load_gpqa_diamond(gpqa_csv)
    append('gpqa', gpqa, prompts.MCQ_SYSTEM)
    append('scibench', knowledge_bench._load_scibench(), prompts.SCIENCE_SYSTEM)
    append('mbpp', code_bench._load_mbpp('full'), prompts.CODE_SYSTEM)
    append('humaneval', code_bench._load_humaneval(), prompts.CODE_SYSTEM)
    for task in ('logical_deduction', 'object_counting', 'tracking_shuffled_objects'):
        append(task, bbh_logic.load_family(task), prompts.BBH_LOGIC_SYSTEM)
    validate_items(items)
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpqa_csv', default=None)
    args = parser.parse_args()
    os.umask(0o077)
    items = load_items(args.gpqa_csv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'items.jsonl').open('x', encoding='utf-8') as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    write_json(output / 'data_manifest.json', {
        'profile': PROFILE_ID, 'n_examples': len(items),
        'portable_data_fingerprint': DATA_FINGERPRINT,
        'mmlu_revision': 'b189ec765aa7ed75c8acfea42df31fdae71f97be',
        'gpqa_access': 'authorized HF access or user-provided CSV; not bundled',
    })
    print(f'DATA_VALIDATED {len(items)} {DATA_FINGERPRINT}')


if __name__ == '__main__':
    main()
