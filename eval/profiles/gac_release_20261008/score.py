"""CPU-only reasoning scoring and final aggregation; no Python execution."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from final_code_protocol import final_code_text
from protocol import (DATA_FINGERPRINT, PROFILE_ID, aggregate, read_jsonl,
                      validate_items, validate_predictions, write_json)
from reasoning import score


def prepare_scores(output):
    if importlib.metadata.version('math-verify') != '0.9.0':
        raise RuntimeError('This profile requires math-verify==0.9.0')
    if not (output / 'GENERATION_COMPLETE').is_file():
        raise RuntimeError('Generation did not finish')
    manifest = json.loads((output / 'generation_manifest.json').read_text())
    if manifest['profile'] != PROFILE_ID or manifest['portable_data_fingerprint'] != DATA_FINGERPRINT:
        raise ValueError('Wrong generation profile or data fingerprint')
    items = read_jsonl(output / 'items.jsonl')
    validate_items(items)
    by_id = {(row['benchmark'], row['id']): row['eval_item'] for row in items}
    rows = read_jsonl(output / 'predictions.jsonl')
    validate_predictions(rows, items)
    scored = []
    with (output / 'code_requests.jsonl').open('x', encoding='utf-8') as handle:
        for row in rows:
            task = row['benchmark']
            item = by_id[(task, row['id'])]
            if task in ('mbpp', 'humaneval'):
                request = {**row, 'eval_item': item,
                           'completion': final_code_text(row['completion'])}
                handle.write(json.dumps(request, ensure_ascii=False) + '\n')
            else:
                scored.append({key: row[key] for key in ('mode', 'benchmark', 'id')})
                scored[-1]['correct'] = bool(score(task, row['completion'], item))
    write_json(output / 'reasoning_scores.json', scored)
    print(f'REASONING_SCORED {len(scored)}; code_requests prepared, not executed')


def finalize(output):
    items = read_jsonl(output / 'items.jsonl')
    validate_items(items)
    reasoning = json.loads((output / 'reasoning_scores.json').read_text())
    code = json.loads((output / 'code_scores.json').read_text())
    if any(row.get('isolated') is not True or row.get('timeout_seconds') != 3
           for row in code):
        raise ValueError('Code scores are not from the declared isolated 3-second protocol')
    rows = reasoning + code
    expected = {(f'seed_{seed}', item['benchmark'], item['id'])
                for seed in (0, 1, 2) for item in items}
    keys = [(row['mode'], row['benchmark'], row['id']) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Incomplete, duplicated, or wrong task scores')
    result = aggregate(rows)
    write_json(output / 'results.json', result)
    for name, metric in result['metrics'].items():
        print(f"{name:28s} {metric['mean_percentage']:.2f}% (three-seed mean)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'finalize'])
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    function = prepare_scores if args.stage == 'prepare' else finalize
    function(Path(args.output_dir))


if __name__ == '__main__':
    main()
