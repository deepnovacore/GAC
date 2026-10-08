"""Joint-batch GPU generation only; never executes generated Python."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

from protocol import (DATA_FINGERPRINT, GENERATION, PROFILE_ID, SEEDS,
                      read_jsonl, validate_items, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_revision', default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--tp_size', type=int, choices=[4], default=4)
    args = parser.parse_args()
    os.umask(0o077)
    # Match the measured sampler, rather than selecting a different kernel.
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    output = Path(args.output_dir)
    if (output / 'predictions.jsonl').exists():
        raise FileExistsError('Use a new output_dir; predictions are never overwritten')
    items = read_jsonl(output / 'items.jsonl')
    validate_items(items)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, revision=args.model_revision, trust_remote_code=False)
    rendered = [tokenizer.apply_chat_template(
        row['messages'], tokenize=False, add_generation_prompt=True) for row in items]
    if any(len(tokenizer.encode(prompt)) + GENERATION['max_new_tokens'] >
           GENERATION['max_model_len'] for prompt in rendered):
        raise ValueError('Prompt exceeds the frozen context budget; no silent truncation')
    llm = LLM(model=args.model_path, revision=args.model_revision,
              tensor_parallel_size=args.tp_size, dtype=GENERATION['dtype'],
              max_model_len=GENERATION['max_model_len'],
              gpu_memory_utilization=GENERATION['gpu_memory_utilization'],
              trust_remote_code=False, seed=GENERATION['engine_seed'],
              enable_prefix_caching=False, enable_chunked_prefill=True)
    with (output / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
        for seed in SEEDS:
            start = time.monotonic()
            generated = llm.generate(rendered, SamplingParams(
                temperature=GENERATION['temperature'], top_p=GENERATION['top_p'],
                max_tokens=GENERATION['max_new_tokens'], n=1, seed=seed))
            if len(generated) != len(items):
                raise RuntimeError('Incomplete generation batch')
            for item, prompt, response in zip(items, rendered, generated):
                if response.prompt != prompt or len(response.outputs) != 1:
                    raise RuntimeError('Generation order or candidate-count drift')
                completion = response.outputs[0]
                handle.write(json.dumps({
                    'benchmark': item['benchmark'], 'id': item['id'],
                    'mode': f'seed_{seed}', 'completion': completion.text,
                    'output_tokens': len(completion.token_ids),
                    'finish_reason': completion.finish_reason,
                    'stop_reason': completion.stop_reason,
                    'prompt_tokens': len(response.prompt_token_ids),
                }, ensure_ascii=False) + '\n')
            handle.flush()
            print(f'COMPLETE seed_{seed} {len(items)} {time.monotonic()-start:.1f}s', flush=True)
    write_json(output / 'generation_manifest.json', {
        'profile': PROFILE_ID, 'portable_data_fingerprint': DATA_FINGERPRINT,
        'n_examples': len(items), 'seeds': list(SEEDS), **GENERATION,
        'versions': {key: importlib.metadata.version(key)
                     for key in ('vllm', 'torch', 'transformers', 'datasets')},
        'model_revision': args.model_revision,
        'batching': 'one joint 4447-question batch per decoding seed',
    })
    (output / 'GENERATION_COMPLETE').touch(exist_ok=False)


if __name__ == '__main__':
    main()
