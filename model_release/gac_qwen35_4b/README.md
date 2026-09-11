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

## Noise-Aware Adaptive Mixing for Hybrid SFT–RL Post-Training

This is the public compact-backbone release of **GAC**, a closed-form,
noise-aware controller that adaptively mixes supervised fine-tuning (SFT) and
reinforcement learning (RL) signals during post-training.

The release is designed for researchers who want to study adaptive SFT–RL
post-training with a contemporary, accessible model backbone. The algorithmic
implementation, evaluation harness, and method description are available in
the [GAC repository](https://github.com/huyuelin/GAC).

## Why Qwen3.5-4B instead of Qwen2.5-7B?

The original paper reports its main results with Qwen2.5-7B-Instruct. This
release deliberately uses **Qwen3.5-4B**, a contemporary frontier compact
backbone, to test whether GAC transfers to a newer and substantially smaller
model class. This makes the release more relevant to accessible post-training
and deployment settings. It is a compact-backbone release, not a claim that
the original Qwen2.5-7B experiment has been rerun with identical weights,
compute, seeds, or evaluation settings.

## Model details

- **Base model:** Qwen3.5-4B-Base
- **Post-training method:** GAC (hybrid SFT–RL adaptive mixing)
- **Parameter scale:** approximately 4B language-model parameters
- **Format:** Hugging Face Transformers / `safetensors`
- **Modalities:** the checkpoint retains the Qwen3.5 text-and-vision model
  interface; the reported evaluation is text-only reasoning and coding
- **License:** Apache-2.0, subject to the upstream Qwen3.5 license and notices

The public repository contains inference weights and configuration only. It
does not contain optimizer states, private training logs, benchmark prompts or
predictions, cluster paths, credentials, or other internal operational data.

## Evaluation snapshot

The table below reports the compact-backbone evaluation snapshot used for this
release, alongside the paper reference and the Qwen3.5-4B base checkpoint.
Percentages are reported as supplied by the evaluation record.

| Domain | Benchmark | Paper GAC + Token-φ | Qwen3.5-4B Base | GAC (Ours) |
|:--|:--|--:|--:|--:|
| Mathematics | AMC | 67.2% | 19.3% | **67.3%** |
| Mathematics | AIME24 | 20.8% | 0.0% | **26.7%** |
| Mathematics | AIME25 | 19.8% | 3.3% | **19.8%** |
| Knowledge | MMLU-Pro | 58.6% | 58.3% | **74.2%** |
| Science | GPQA-Diamond | 43.5% | 29.3% | **64.1%** |
| Science | SciBench | 41.2% | 11.4% | **60.1%** |
| Code | MBPP | 78.8% | 67.6% | **74.6%** |
| Code | HumanEval | 83.5% | 67.7% | **81.3%** |
| Logic | BBH Logical Deduction | 66.5% | 86.3% | **93.1%** |
| Logic | BBH Object Counting | 70.9% | 93.2% | **92.8%** |
| Logic | BBH Tracking | 59.7% | 97.6% | **95.3%** |
| Logic | BBH average | 65.7% | 92.4% | **90.4%** |

The paper column is the original Qwen2.5-7B-Instruct result, reported as a
multi-seed paper reference. The base and GAC columns are compact-backbone
evaluation snapshots. They should therefore be read as a transfer and
accessibility study, not as a strictly paired statistical comparison.

### Reading the result

Relative to the Qwen3.5-4B base snapshot, GAC shows substantial gains on math,
knowledge, science, and code, while BBH remains in a high-performance regime
with small subset-level variation. This pattern is consistent with the
intended role of GAC: adaptively controlling the SFT/RL mixture as the signal
and noise profile changes during post-training.

## Quick start

### Transformers

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "YueLinHu/GAC-Qwen3.5-4B"  # replace the namespace for other mirrors

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "user", "content": [{"type": "text", "text": "Solve: 2 + 2 = ?"}]}
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(text=[text], return_tensors="pt").to(model.device)
with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=256)
print(processor.batch_decode(output_ids, skip_special_tokens=True)[0])
```

Use the official Qwen3.5 documentation for multimodal inputs and the
transformers version required by the checkpoint. For text-only benchmark
reproduction, follow the commands in the repository's
[`eval/README.md`](https://github.com/huyuelin/GAC/tree/main/eval).

## Reproducibility

The algorithmic implementation, default controller configuration, unit tests,
and evaluation harness are available at:

- Code: https://github.com/huyuelin/GAC
- Paper and method description: https://openreview.net/forum?id=VhBpT4iq60
- Evaluation harness: https://github.com/huyuelin/GAC/tree/main/eval

For scientifically comparable reporting, record the exact model revision,
transformers/vLLM version, decoding configuration, dataset revision, random
seed, pass@1 execution sandbox, and scoring parser. The values above are a
release snapshot rather than a replacement for the paper's multi-seed mean ±
standard deviation protocol.

## Limitations and responsible use

- Benchmark accuracy is not a guarantee of reliability, factuality, or safety
  in deployment.
- Code-generation outputs must be executed in an isolated sandbox.
- The compact-backbone results are not directly interchangeable with the
  original Qwen2.5-7B paper results.
- Users are responsible for complying with the licenses and usage policies of
  Qwen3.5, the evaluation datasets, and any downstream application.

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

## Acknowledgements

This checkpoint builds on Qwen3.5 and the GAC post-training method. Please
cite both the GAC paper and the upstream Qwen3.5 model when appropriate.
