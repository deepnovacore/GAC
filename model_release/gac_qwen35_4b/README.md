---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE
base_model:
- Qwen/Qwen3.5-4B-Base
pipeline_tag: image-text-to-text
tags:
- gac
- qwen3.5
- post-training
- reinforcement-learning
- supervised-fine-tuning
- reasoning
- text-generation
- vision-language
---

# GAC-Qwen3.5-4B

GAC-Qwen3.5-4B brings noise-aware adaptive SFT–RL post-training to the
Qwen3.5-4B backbone. This release combines GAC post-training with a supervised
mathematics and general-dialogue refinement stage, and provides BF16 inference
weights, a complete three-seed evaluation summary, and a versioned evaluation
profile.

[Paper](https://openreview.net/forum?id=VhBpT4iq60) ·
[Code](https://github.com/deepnovacore/GAC) ·
[Project page](https://deepnovacore.github.io/GAC/) ·
[Release evaluation profile](https://github.com/deepnovacore/GAC/tree/main/eval/profiles/gac_release_20261008)

## Model details

- **Backbone:** Qwen3.5-4B-Base.
- **Post-training:** GAC hybrid SFT–RL followed by full-parameter supervised refinement.
- **Format:** BF16 `safetensors`, with native Qwen3.5 Transformers configuration.
- **Interface:** the text-and-vision architecture is retained; the scores below evaluate text-only reasoning and code.
- **Release:** October 2026, with evaluation performed on October 3, 2026.
- **License:** Apache-2.0, subject to upstream Qwen3.5 notices and dataset terms.

## Evaluation results

The table reports the **mean over decoding seeds 0, 1, and 2** for this
checkpoint: 4,447 questions across 11 task slices, with one generated answer
per question per seed. Code scores are `pass@1`. Exact counts and unrounded
means are available in [EVAL_RESULTS.json](EVAL_RESULTS.json).

| Domain | Benchmark | Eval size per seed | GAC release three-seed mean |
|:--|:--|--:|--:|
| Mathematics | AMC | 83 | **66.7%** |
| Mathematics | AIME24 | 30 | **34.4%** |
| Mathematics | AIME25 | 30 | **26.7%** |
| Knowledge | MMLU-Pro | 1,000 | **72.5%** |
| Science | GPQA-Diamond | 198 | **59.1%** |
| Science | SciBench | 692 | **58.5%** |
| Code | MBPP | 500 | **68.9%** |
| Code | HumanEval | 164 | **86.6%** |
| Logic | BBH Logical Deduction | 750 | **94.4%** |
| Logic | BBH Object Counting | 250 | **87.7%** |
| Logic | BBH Tracking | 750 | **95.6%** |
| Logic | BBH average (macro) | 3 slices / 1,750 | **92.6%** |

BBH macro-average gives equal weight to the three slice accuracies and is
calculated before rounding. The seeds measure decoding variability for one
checkpoint, not three independent training runs. Evaluation informed
checkpoint development; this snapshot is separate from the paper's experiments.

### Evaluation protocol

All task slices use **temperature 0.6, top-p 0.95, max 8,192 new tokens**,
one sample, BF16 inference, a 16,384-token model context, and tensor parallelism
of 4. The profile generates one joint batch of all task slices for each seed.

| Task family | Scoring |
|:--|:--|
| AMC / AIME24 / AIME25 | Final boxed answer with `math-verify==0.9.0` equivalence |
| MMLU-Pro / GPQA-Diamond | Strict extracted answer-letter match |
| SciBench | Numeric-and-units protocol v2, with balanced final boxes and 1% relative tolerance |
| MBPP / HumanEval | Final-answer code extraction, tested MBPP entrypoint, and original tests in an isolated CPU sandbox |
| BBH logic | Explicit final choices for deduction/tracking; normalized answer matching for object counting |

These are the **release profile** settings. The general evaluator's
task-specific defaults use different code and logic budgets; use the linked
release profile when reproducing this table. MMLU-Pro uses a pinned revision
and the checked-in 1,000-ID manifest. GPQA requires dataset access or a
lawfully obtained complete CSV; its data is not bundled.

## Supervised refinement

The refinement stage uses 128 examples: 64 reviewed mathematics examples and
64 general-dialogue replay examples. It applies two epochs of full-parameter
SFT with learning rate `2e-6`, global batch size 32, constant learning-rate
schedule, one warmup update, shuffle seed `20261001`, and single-example
microbatches without packing. The new refinement examples passed the recorded
overlap checks against the fixed evaluation questions. See
[REFINEMENT_METADATA.json](REFINEMENT_METADATA.json) for source fingerprints,
the complete refinement recipe and the scope of these checks.

## Quick start

Install Transformers with native Qwen3.5 support and Accelerate:

```bash
pip install -U "transformers>=5.10.4" accelerate
```

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "YueLinHu/GAC-Qwen3.5-4B"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=False
)
messages = [{"role": "user", "content": [{"type": "text", "text": "Solve: 2 + 2 = ?"}]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=8192,
                             do_sample=True, temperature=0.6, top_p=0.95)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:],
                       skip_special_tokens=True))
```

## Reproducibility and use

The [release profile](https://github.com/deepnovacore/GAC/tree/main/eval/profiles/gac_release_20261008)
contains the prompts, dataset checks, generation and scoring commands, and
parser tests. [EVAL_RESULTS.json](EVAL_RESULTS.json) records all three seeds'
correct counts, dataset/profile fingerprints, and scoring conventions.
The [release manifest](RELEASE_MANIFEST.json) identifies the published weights
and code revision. Prior model revisions remain available in Hub history.

Use this model for research, reasoning experiments, and code-generation
evaluation. Benchmark accuracy does not establish reliability in high-stakes
applications; execute generated code only in an isolated environment and
follow the licenses and access conditions of the model and datasets.

## Citation

```bibtex
@inproceedings{hu2026gac,
  title     = {GAC: Noise-Aware Adaptive Mixing for Hybrid SFT-RL Post-Training},
  author    = {Hu, Yuelin and Liu, Wei and Yu, Zhenbo and Cheng, Zhengxue and Song, Li},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  address   = {Budapest, Hungary},
  url       = {https://openreview.net/forum?id=VhBpT4iq60}
}
```

Please cite the GAC paper and the upstream Qwen3.5 model when appropriate.
