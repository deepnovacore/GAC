<div align="center">

# 🎼 GAC
### Noise-Aware Adaptive Mixing for Hybrid SFT–RL Post-Training

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-blueviolet.svg?style=flat-square)](https://2026.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-forthcoming-b31b1b.svg?style=flat-square)](#)
[![OpenReview](https://img.shields.io/badge/OpenReview-VhBpT4iq60-8c1b13.svg?style=flat-square)](https://openreview.net/forum?id=VhBpT4iq60)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f.svg?style=flat-square)](https://deepnovacore.github.io/GAC/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-GAC--Qwen3.5--4B-yellow.svg?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/YueLinHu/GAC-Qwen3.5-4B)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)

**A closed-form, noise-aware controller that adaptively balances SFT and RL — no hand-tuned μ schedule required.**

<sub>Yuelin Hu¹ · Wei Liu² · Zhenbo Yu¹,³ · Zhengxue Cheng¹ · Li Song¹</sub>
<sub><sup>¹</sup>Shanghai Jiao Tong University &nbsp;·&nbsp; <sup>²</sup>Shanghai Maritime University &nbsp;·&nbsp; <sup>³</sup>Novacore</sub>

<img src="assets/method_figure.png" alt="GAC method figure" width="90%">

</div>

---

## 📚 Contents

- [🔥 News](#-news)
- [💡 What is GAC?](#-what-is-gac)
- [✨ Key Highlights](#-key-highlights)
- [📊 Results](#-results)
- [🚀 Quick Start](#-quick-start)
- [🔍 Evaluation & Release](#-evaluation--release)
- [📐 Method](#-method)
- [🗂️ Repository Layout](#️-repository-layout)
- [📝 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [📮 Contact](#-contact)

---

## 🔥 News

- **[2026/10/08]** 🤗 Refreshed **GAC-Qwen3.5-4B** weights with a full three-decoding-seed evaluation and a [versioned reproduction profile](eval/profiles/gac_release_20261008/).
- **[2026/09]** 🤗 **GAC-Qwen3.5-4B** checkpoint released on Hugging Face with an evaluation snapshot and model card.
- **[2026/09]** 📊 Public checkpoint evaluator released with dataset provenance, fixed-subset validation, and isolated code scoring.
- **[2026/08]** 📄 Camera-ready released; the reference implementation was open-sourced under Apache-2.0.
- **[2026/08]** 🎉 **GAC was accepted to the EMNLP 2026 Main Conference (Budapest).**

---

## 💡 What is GAC?

Hybrid SFT–RL post-training is the standard recipe for aligning LLMs — but the *mixing weight* **μ** between the two signals is almost always a **fixed, hand-tuned schedule**. This is fragile: when RL noise spikes, or when the SFT expert starts pulling the policy in a wrong direction, a static μ over-commits to whichever side is currently *less* reliable.

**GAC solves this by treating μ as an estimation problem.**

We derive a closed-form optimal μ that minimizes the MSE of the composite gradient under a signal-vs-noise decomposition:

<div align="center">

**μ\* = ( α<sub>tgt</sub>·Δg² + σ<sub>r</sub>² ) / ( Δg² + σ<sub>s</sub>² + σ<sub>r</sub>² )**

</div>

where **σ<sub>s</sub>², σ<sub>r</sub>²** are SFT and RL noise variances and **Δg²** is the SFT–RL disagreement. Since gradient-level quantities are prohibitively expensive at every step, we deploy three **coefficient-space proxies** estimated online from tensors any GRPO/PPO trainer already computes, wrapped in EMA smoothing, a cosine-schedule prior, and per-step change capping.

**Paper result**: **+3.8 pp over HPT** (the previous best hybrid post-training method) averaged over math, code, science, and logic benchmarks, with **< 1% wall-time overhead**, **28% lower KL-drift area**, and consistent gains across the evaluated 1.5B → 14B models.

---

## ✨ Key Highlights

- 🎯 **Closed-form μ\***, derived from MSE-optimal gradient estimation — no meta-learning, no bilevel optimization, no extra hyperparameter grid.
- 📉 **Coefficient-space proxies**: three cheap signals (`σ_s²`, `σ_r²`, `Δg̃²`) that any PPO/GRPO trainer already has on hand.
- 🛡️ **Guarded controller stack**: EMA smoothing → cosine-prior blend → per-step change cap → hard clipping. Robust to mid-training regime shifts.
- ⚡ **< 1% wall-time overhead**: proxies are computed inside the existing forward pass, no extra network eval.
- 📈 **Consistent gains across scales**: **+2.2 pp @ 1.5B**, **+3.8 pp @ 7B**, **+3.3 pp @ 14B** on AMC vs. HPT (paper Table 3).
- 🔬 **Length-invariant by construction**: each sequence contributes exactly one scalar per proxy, so longer rollouts don't get extra vote weight.

---

## 📊 Results

### Qwen3.5-4B compact release

We bring GAC to **Qwen3.5-4B**, combining hybrid SFT–RL post-training with a
reviewed mathematics and general-dialogue supervised refinement stage. The
October release includes BF16 weights and the following **mean over decoding
seeds 0, 1, and 2** for one checkpoint, evaluated on October 3.

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

Each seed generates one answer per question: 4,447 questions and 13,341
generations in total. All slices use temperature 0.6, top-p 0.95 and an
8,192-token output budget in a joint batch per seed; code scores are pass@1.
BBH macro-average weights the three slices equally before rounding.
These decoding seeds are not independent training runs. Evaluation informed
checkpoint development; the release is separate from the paper experiments.

See the [release profile](eval/profiles/gac_release_20261008/) for exact
prompts, parsers and commands, and
[EVAL_RESULTS.json](model_release/gac_qwen35_4b/EVAL_RESULTS.json) for every
seed's counts. General evaluator defaults differ from this profile.

For the original paper's experiments and tables, see the
[OpenReview paper](https://openreview.net/forum?id=VhBpT4iq60).

---

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/deepnovacore/GAC.git
cd GAC
pip install -r requirements.txt
pip install -e .
```

The controller has **no framework-specific dependencies** beyond PyTorch. `torch.distributed` is used only if you have multi-GPU.

### Minimal integration (5 lines)

Drop the controller into any hybrid SFT–RL loop:

```python
from gac import AdaptiveMuController
from gac.sft_loss import SFTPhiLossFn
from gac.hybrid_loss import compute_hybrid_loss
from gac.controller import mu_schedule

controller = AdaptiveMuController(
    ema_beta=0.99,
    mu_change_cap=0.01,
    trim_ratio_sft=0.10,
    mu_update_freq=10,
    alpha_static=0.5,
)
sft_loss_fn = SFTPhiLossFn(token_level=True)

def training_step(batch, global_step, rl_loss_fn):
    # cosine-schedule prior on μ (peak → valley over training)
    prior = mu_schedule(
        global_step,
        warmup_steps=200, decay_steps=800,
        mu_peak=0.85, mu_valley=0.15,
    )
    # μ is inferred inside compute_hybrid_loss via the controller
    loss, logs = compute_hybrid_loss(
        logprob=batch["logprob"],
        action_mask=batch["action_mask"],
        advantages=batch["advantages"],
        expert_mask=batch["expert_mask"],
        rl_loss_fn=rl_loss_fn,          # your favorite PPO / GRPO surrogate
        sft_loss_fn=sft_loss_fn,
        controller=controller,
        global_step=global_step,
        schedule_prior=prior,
        blend_weight=0.5,               # λ in the paper — cosine-prior blend
    )
    return loss, logs
```

That's it. GAC only touches the **weighting between** the SFT and RL losses; it does not modify either loss itself. You keep your PPO / GRPO implementation, your RL infrastructure, and your reward model unchanged.

---

## 🔍 Evaluation & Release

The public [`eval/`](eval/) harness covers **11 task slices across 9 dataset
families**: AMC, AIME24, AIME25, MMLU-Pro, GPQA-Diamond, SciBench, MBPP,
HumanEval, and three BBH logic subsets. It includes dataset provenance,
fixed-subset validation, aggregation, and isolated code scoring.

```bash
cd eval
pip install -r requirements.txt
bash run_all.sh --model_path /path/to/your-checkpoint \
  --output_dir ./results --tp_size 4
```

MMLU-Pro uses a pinned revision and a checked-in 1,000-ID manifest. GPQA is a
gated dataset: authenticate with Hugging Face or provide a complete, lawfully
obtained CSV through `--gpqa_csv`. See [`eval/README.md`](eval/README.md) for
benchmark commands, scoring details, Slurm usage, and the code-execution
safety boundary.

To reproduce the current model-card table, use the
[October release profile](eval/profiles/gac_release_20261008/) instead of
the general task-specific defaults above.

The [GAC-Qwen3.5-4B checkpoint](https://huggingface.co/YueLinHu/GAC-Qwen3.5-4B)
contains the compact-backbone release and its evaluation protocol. The
original paper's experimental tables remain available in the
[OpenReview paper](https://openreview.net/forum?id=VhBpT4iq60).

### 🗺️ Roadmap

| Version | Target | Contents |
|---|---|---|
| ✅ **v0.1.0** | 2026-08 | Reference `AdaptiveMuController`, hybrid-loss integration, unit tests, and evaluator covering 11 task slices |
| ✅ **v0.1.1** | 2026-09 | Fixed-subset validation, isolated code scoring, provenance checks, and Qwen3.5-4B checkpoint release |
| ✅ **v0.1.2** | 2026-10 | Refreshed compact model, complete three-seed results, frozen evaluation profile, and parser tests |
| 🚧 **Planned** | — | Training-infrastructure packaging, ModelScope mirror, container recipe, and external verification |

---

## 📐 Method

The controller performs three operations per update:

**1. Proxy estimation** *(Sec. 3.2 of the paper)*

| Symbol | Meaning | Length-invariant? |
|---|---|:---:|
| σ<sub>r</sub>² | Variance of *sequence-level* GRPO-normalized advantages across the batch. **Post-normalization dispersion**, not raw reward variance. | ✅ |
| σ<sub>s</sub>² | Tail-trimmed variance of length-normalized per-sequence NLL on expert samples. | ✅ |
| Δg̃² | Mean squared coefficient mismatch between SFT and RL token-level gradient coefficients on shared response tokens, after within-batch z-normalization. | ✅ |

**2. Closed-form aggregation** into μ\*:

<div align="center">

μ\* = ( α<sub>tgt</sub>·Δg̃² + σ<sub>r</sub>² ) / ( Δg̃² + σ<sub>s</sub>² + σ<sub>r</sub>² )

</div>

**3. Guarded update**:

```
raw_μ*  →  EMA(β)  →  blend(λ) with cosine prior  →  clip |Δμ| ≤ c̄  →  clip to [μ_min, μ_max]
```

Detailed derivation (including the MSE minimization, the proxy-vs-oracle validation, and length-invariance proofs) is in **[docs/method.md](docs/method.md)** and Appendices A–D of the paper.

---

## 🗂️ Repository Layout

```
GAC/
├── gac/                     # Reference implementation
│   ├── controller.py        # AdaptiveMuController (paper Sec. 3)
│   ├── sft_loss.py          # SFT loss + CHORD φ(p)=p(1-p) variant
│   ├── hybrid_loss.py       # Reference hybrid loss (μ·L_SFT + (1-μ)·L_RL)
│   ├── utils.py             # Length-invariant masked reductions
│   └── __init__.py
├── eval/                    # Public checkpoint evaluation harness
│   ├── generate_vllm.py     # Unified vLLM generator
│   ├── math_bench.py        # AMC / AIME24 / AIME25 (math-verify scoring)
│   ├── knowledge_bench.py   # MMLU-Pro / GPQA / SciBench
│   ├── code_bench.py        # MBPP / HumanEval (bigcode-eval scoring)
│   ├── bbh_logic.py         # BBH logical subsets
│   ├── aggregate.py         # Multi-seed mean±std aggregation
│   ├── manifests/            # Pinned benchmark subset manifests
│   ├── run_all.sh           # One-command reproduction
│   └── README.md            # Detailed usage & data provenance
├── configs/
│   └── gac_default.yaml     # Default hyperparameters (paper Sec. 4.1)
├── docs/
│   ├── method.md            # Long-form derivation and proxy semantics
│   └── index.html           # Project page (served by GitHub Pages)
├── tests/
│   ├── test_controller.py   # Closed-form estimator unit tests
│   └── test_evaluation_data.py # Dataset and fixed-subset regression tests
├── assets/                  # Figures used in this README
├── CITATION.cff
├── LICENSE                  # Apache 2.0
├── README.md
├── requirements.txt
└── setup.py
```

---

## 📝 Citation

If you use GAC in your research, please cite:

```bibtex
@inproceedings{hu2026gac,
  title     = {GAC: Noise-Aware Adaptive Mixing for Hybrid SFT-RL Post-Training},
  author    = {Hu, Yuelin and Liu, Wei and Yu, Zhenbo and Cheng, Zhengxue and Song, Li},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  address   = {Budapest, Hungary},
  url       = {https://openreview.net/forum?id=VhBpT4iq60},
}
```

Machine-readable metadata is also provided in [`CITATION.cff`](CITATION.cff).

---

## 🙏 Acknowledgements

We thank **Novacore** for their collaboration on this work. This work is a **joint academic collaboration with Novacore**; no commercial rights are asserted.

This research was supported in part by the NSFC (62431015, 62571317, 62501387), the Fundamental Research Funds for the Central Universities, Shanghai Key Laboratory of Digital Media Processing and Transmission under Grant 22DZ2229005, 111 project BP0719010, and the **Ant Group Research Fund** (academic grant; no IP claim).

The GAC design builds on ideas explored by prior hybrid SFT–RL literature — in particular the token-wise reweighting introduced by [CHORD](https://arxiv.org/abs/2508.11408) and the hybrid post-training framework of [HPT](https://arxiv.org/abs/2509.04419). Our reference implementation integrates cleanly into GRPO / PPO trainers such as [VeRL](https://github.com/volcengine/verl) and [Trinity-RFT](https://github.com/modelscope/Trinity-RFT). We thank the anonymous ARR reviewers and the EMNLP 2026 Area Chair for the detailed feedback that shaped the camera-ready version of the paper.

---

## 📮 Contact

For questions, feedback, or collaboration opportunities:

- **Yuelin Hu** — <huyuelin51717221@sjtu.edu.cn> *(first author, code maintainer)*
- **Zhenbo Yu** — <yuzhenbo@sjtu.edu.cn> *(co-corresponding, Novacore)*
- **Li Song** — <songli@sjtu.edu.cn> *(co-corresponding, SJTU)*

**Issues & PRs are very welcome.** If you're integrating GAC into a specific trainer (TRL, OpenRLHF, VeRL, Trinity-RFT, custom) and hit rough edges, please open an issue with your integration snippet — it directly helps others.

---

<div align="center">
<sub>Released under the <a href="LICENSE">Apache License 2.0</a>. Copyright © 2026 the GAC authors.</sub>
</div>
