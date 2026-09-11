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
- **Transformers compatibility:** use Transformers 5.10.4 or newer with native
  Qwen3.5 support
- **Modalities:** the checkpoint retains the Qwen3.5 text-and-vision model
  interface; the reported evaluation is text-only reasoning and coding
- **License:** Apache-2.0, subject to the upstream Qwen3.5 license and notices

The public repository contains inference weights and configuration only. It
does not contain optimizer states, private training logs, benchmark prompts or
predictions, cluster paths, credentials, or other internal operational data.

## Intended use

This checkpoint is intended for research on adaptive hybrid SFT–RL
post-training, reproducible checkpoint evaluation, and low-cost experimentation
with compact multimodal language models. It is not a safety-certified system
and should not be used for unsupervised high-stakes decisions.

## Training and provenance

This is an inference checkpoint release. The public GAC implementation and
evaluation harness document the method and evaluation procedure, while the
private run artifacts (optimizer state, rollout traces, training logs, and
internal storage paths) are not part of the release. Consequently, this card
supports inference and checkpoint-level evaluation; it does not claim that the
exact training run can be reconstructed from the model repository alone.

## Evaluation snapshot

The table below reports the compact-backbone evaluation snapshot for the base
checkpoint and the GAC release checkpoint. Percentages are reported as supplied
by the evaluation record; a rerun with the public evaluator writes exact
per-example counts and summaries.

| Domain | Benchmark | Eval size (N) | Qwen3.5-4B Base | GAC release |
|:--|:--|--:|--:|--:|
| Mathematics | AMC | 83 | 19.3% | **67.3%** |
| Mathematics | AIME24 | 30 | 0.0% | **26.7%** |
| Mathematics | AIME25 | 30 | 3.3% | **19.8%** |
| Knowledge | MMLU-Pro | 1,000 | 58.3% | **74.2%** |
| Science | GPQA-Diamond | 198 | 29.3% | **64.1%** |
| Science | SciBench | 692 | 11.4% | **60.1%** |
| Code | MBPP | 500 | 67.6% | **74.6%** |
| Code | HumanEval | 164 | 67.7% | **81.3%** |
| Logic | BBH Logical Deduction | 750 | 86.3% | **93.1%** |
| Logic | BBH Object Counting | 250 | 93.2% | **92.8%** |
| Logic | BBH Tracking | 750 | 97.6% | **95.3%** |
| Logic | BBH average (macro) | 3 slices / 1,750 | 92.4% | **93.7%** |

The original paper uses Qwen2.5-7B-Instruct, whereas this release evaluates a
Qwen3.5-4B compact backbone. The two columns above should therefore be read as
a same-backbone base-versus-GAC snapshot, not as a multi-seed statistical
comparison or a like-for-like replacement of the paper experiment. No integer
success count is inferred from rounded percentages in this card.

### Reading the result

Relative to the Qwen3.5-4B base snapshot, the GAC release is higher on 9 of 11
task slices in this snapshot, with gains across math, knowledge, science, and
code. BBH remains in a high-performance regime: the macro-average is higher,
while Object Counting and Tracking are slightly below the base snapshot. This
is descriptive evidence for transfer to a compact backbone, not a claim of
universal improvement.

### Evaluation protocol

For an apples-to-apples rerun, use the public evaluator with `--n_samples 1`
and record the model revision, software versions, dataset revision, and seed.
The harness uses the following task-specific defaults:

| Task family | Decoding | Scoring |
|:--|:--|:--|
| AMC / AIME24 / AIME25 | temperature 0.6, top-p 0.95, max 8,192 tokens | `math-verify` equivalence |
| MMLU-Pro / GPQA-Diamond | temperature 0.6, top-p 0.95, max 8,192 tokens | strict answer-letter match |
| SciBench | temperature 0.6, top-p 0.95, max 8,192 tokens | SymPy equivalence plus numeric fallback |
| MBPP / HumanEval | temperature 0.2, top-p 0.95, max 1,024 tokens | isolated `pass@1` execution |
| BBH logical subsets | temperature 0.6, top-p 0.95, max 4,096 tokens | normalized exact match |

Dataset IDs, splits, fixed-subset selection, and the required example counts
are maintained in [`eval/README.md`](https://github.com/huyuelin/GAC/tree/main/eval).
GPQA-Diamond is gated on Hugging Face; the evaluator accepts the public CSV
mirror through `--gpqa_csv` and refuses partial datasets.

## Quick start

### Transformers

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "YueLinHu/GAC-Qwen3.5-4B"  # replace the namespace for other mirrors

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
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

Install a current Transformers release with native Qwen3.5 support before
loading the checkpoint, for example `pip install -U "transformers>=5.10.4"`.
Use the official Qwen3.5 documentation for multimodal inputs. For text-only
benchmark reproduction, follow the commands in the repository's
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
fixed release snapshot rather than multi-seed confidence intervals. When
`n_samples` is greater than one, the harness reports best-of-n for generated
answers; use `n_samples=1` for the pass@1-style snapshot shown above.

## Limitations and responsible use

- Benchmark accuracy is not a guarantee of reliability, factuality, or safety
  in deployment.
- Code-generation outputs must be executed in an isolated sandbox.
- The compact-backbone results are not directly interchangeable with the
  original Qwen2.5-7B experiment and are not multi-seed confidence intervals.
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
