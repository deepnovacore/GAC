"""Original BigCode tests, only inside the offline isolation wrapper."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from code_protocol import normalize_candidate
from protocol import SEEDS, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--bigcode_root', required=True)
    args = parser.parse_args()
    if os.environ.get('GAC_CODE_EVAL_SANDBOX') != '1' or os.environ.get('HF_DATASETS_OFFLINE') != '1':
        raise RuntimeError('Refusing code execution outside the offline namespace wrapper')
    sys.path.insert(0, args.bigcode_root)
    from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
    output = Path(args.output_dir)
    if (output / 'code_scores.json').exists():
        raise FileExistsError('Code scores are never overwritten')
    rows = read_jsonl(output / 'code_requests.jsonl')
    keys = [(row['mode'], row['benchmark'], row['id']) for row in rows]
    if len(rows) != 1992 or len(set(keys)) != 1992:
        raise ValueError('Expected 3 x (500 MBPP + 164 HumanEval) unique code requests')
    for seed in SEEDS:
        for task, count in [('mbpp', 500), ('humaneval', 164)]:
            if sum(row['mode'] == f'seed_{seed}' and row['benchmark'] == task for row in rows) != count:
                raise ValueError('Wrong code task/seed counts')
    programs, references, modes = [], [], []
    for row in rows:
        text = row['completion']
        match = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
        candidate = match[1].strip() if match else text.strip()
        context = dict(row['eval_item'])
        if row['benchmark'] == 'mbpp':
            context['test'] = '\n'.join(context['test_list'])
        try:
            program, mode = normalize_candidate(row['benchmark'], candidate, context)
        except (SyntaxError, ValueError, IndentationError) as error:
            program = 'raise AssertionError("candidate normalization failed")\n'
            mode = 'invalid_' + type(error).__name__
        programs.append([program])
        modes.append(mode)
        references.append(context['test'] + (
            '\ncheck(' + context['entry_point'] + ')' if row['benchmark'] == 'humaneval' else ''))
    os.environ['HF_ALLOW_CODE_EVAL'] = '1'
    _, details = compute_code_eval(predictions=programs, references=references,
                                  k=[1], num_workers=8, timeout=3.)
    result = []
    for index, row in enumerate(rows):
        if len(details[index]) != 1:
            raise RuntimeError('Executor returned an unexpected candidate count')
        _, answer = details[index][0]
        result.append({key: row[key] for key in ('mode', 'benchmark', 'id')})
        result[-1].update(correct=bool(answer['passed']), normalization=modes[index],
                          execution_result=str(answer['result']), timeout_seconds=3, isolated=True)
    write_json(output / 'code_scores.json', result)
    print(f'ISOLATED_CODE {sum(row["correct"] for row in result)} / {len(result)}')


if __name__ == '__main__':
    main()
