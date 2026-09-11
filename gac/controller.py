"""Guided Adaptive Controller for hybrid SFT-RL post-training.

This module implements :class:`AdaptiveMuController`, the noise-aware mixing
controller introduced in

    Hu, Yu, Liu, Cheng, and Song.
    *GAC: Noise-Aware Adaptive Mixing for Hybrid SFT-RL Post-Training.*
    EMNLP 2026 (Main Conference).

The controller computes the global mixing weight :math:`\\mu \\in [0, 1]`
between the supervised anchoring loss and the on-policy RL loss by solving a
local mean-squared-error objective over the mixed stochastic gradient. Under
the unbiased, variance-dominated regime studied in the paper (Sec. 3), the
closed-form optimum is

.. math::

    \\mu^* = \\frac{\\alpha_\\mathrm{tgt}\\,\\Delta g^2 + \\sigma_r^2}
                     {\\Delta g^2 + \\sigma_s^2 + \\sigma_r^2},

where :math:`\\sigma_s^2, \\sigma_r^2` are the SFT and RL gradient-noise
variances and :math:`\\Delta g^2` is the SFT--RL gradient disagreement. In
deployment, the three inputs are replaced by coefficient-space proxies
(Sec. 3.2) and the raw estimate is stabilised by EMA smoothing, cosine-prior
blending, and per-step change capping (Sec. 3.4), producing the update rule
implemented in :meth:`AdaptiveMuController.compute_mu`.

Notes
-----
The proxies consumed by :meth:`AdaptiveMuController.update_stats` are:

* :math:`\\sigma_r^2` -- variance of *sequence-level* GRPO advantages after
  the standard within-group normalisation, i.e. a post-normalisation
  coefficient-dispersion estimate rather than raw reward variance.
* :math:`\\sigma_s^2` -- length-normalised, tail-trimmed variance of the
  per-sequence negative log-likelihood on expert samples.
* :math:`\\Delta\\tilde g^2` -- squared mismatch of the token-level gradient
  *coefficients* on shared response positions; masking and normalisation are
  applied inside :meth:`update_stats`.

The controller is length-invariant by construction: each per-sequence
statistic contributes exactly one scalar to the batch aggregate, so longer
rollouts do not receive disproportionate weight (see Sec. 4 of the paper).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from gac.utils import masked_mean


class AdaptiveMuController:
    """Online estimator of the mixing weight :math:`\\mu` for hybrid SFT-RL.

    Parameters
    ----------
    ema_beta : float, default 0.99
        EMA smoothing coefficient. Larger values impose stronger low-pass
        filtering on the raw closed-form estimate.
    mu_min, mu_max : float
        Hard bounds for :math:`\\mu` after each update. Values are clipped so
        that neither the SFT nor the RL branch is fully suppressed.
    mu_update_freq : int, default 10
        Number of optimiser steps between successive updates of the
        underlying EMA statistics. Throttling reduces mini-batch noise while
        keeping wall-time overhead below one percent of total training time.
    alpha_static : float, default 0.5
        Fallback value of the KL-controlled target ratio
        :math:`\\alpha_\\mathrm{ctrl}` when the outer training loop does not
        supply a dynamic override.
    use_phi_in_sft : bool, default True
        If ``True``, the SFT gradient coefficient in the disagreement proxy
        is taken as :math:`\\varphi(p) = p(1-p)` (the CHORD reweighting);
        otherwise a constant unit coefficient is used.
    init_mu : float, default 0.5
        Initial value of :math:`\\mu`. Used before any EMA statistics have
        been accumulated.
    mu_change_cap : float, default 0.01
        Per-update change cap :math:`\\bar c` (Sec. 3.4). Limits the maximum
        absolute jump in :math:`\\mu` between consecutive controller calls
        to guarantee small-step behaviour analogous to PPO surrogate
        clipping.
    trim_ratio_sft : float, default 0.1
        Two-sided tail-trim ratio for :math:`\\sigma_s^2`. Robustifies the
        NLL variance against a small number of anomalously long or degenerate
        expert samples.
    delta_g2_cap_ratio : float, default 0.0
        Optional cap on :math:`\\Delta\\tilde g^2` expressed as a fraction of
        :math:`\\sigma_s^2 + \\sigma_r^2`. Zero disables the cap; positive
        values prevent transient disagreement spikes from dominating the
        estimator numerator during early training.
    eps : float, default 1e-8
        Numerical stabilisation constant for variance clamping and division.
    """

    def __init__(
        self,
        ema_beta: float = 0.99,
        mu_min: float = 0.05,
        mu_max: float = 0.9,
        mu_update_freq: int = 10,
        alpha_static: float = 0.5,
        use_phi_in_sft: bool = True,
        init_mu: float = 0.5,
        mu_change_cap: float = 0.01,
        trim_ratio_sft: float = 0.1,
        delta_g2_cap_ratio: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        self.ema_beta = ema_beta
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_update_freq = mu_update_freq
        self.alpha_static = alpha_static
        self.use_phi_in_sft = use_phi_in_sft
        self.mu = init_mu
        self.mu_change_cap = mu_change_cap
        self.eps = eps
        self.trim_ratio_sft = trim_ratio_sft
        self.delta_g2_cap_ratio = delta_g2_cap_ratio

        # EMA statistics; initialised lazily on the first update.
        self.ema_delta_g2: Optional[float] = None
        self.ema_sigma_s2: Optional[float] = None
        self.ema_sigma_r2: Optional[float] = None

        # Step counter used to throttle updates.
        self._last_update_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _masked_var(
        x: torch.Tensor,
        mask: torch.Tensor,
        axis: Optional[int] = None,
    ) -> torch.Tensor:
        """Numerically stable masked variance via :math:`E[X^2] - E[X]^2`."""
        ex = masked_mean(x, mask, axis=axis)
        ex2 = masked_mean(x * x, mask, axis=axis)
        return torch.clamp(ex2 - ex * ex, min=0.0)

    @staticmethod
    def _trimmed_variance(
        values: torch.Tensor,
        trim_ratio: float,
        eps: float,
    ) -> float:
        """Two-sided trimmed variance of a 1-D tensor.

        Drops the top and bottom ``trim_ratio`` fraction of samples before
        computing the (biased) variance. For tensors shorter than four
        elements or ``trim_ratio == 0``, falls back to the untrimmed variance.
        """
        if values.numel() == 0:
            return 0.0
        trim_ratio = float(max(0.0, min(0.49, trim_ratio)))
        if trim_ratio == 0.0 or values.numel() < 4:
            var = torch.var(values, unbiased=False)
            return float(torch.clamp(var, min=eps).item())
        k = int(values.numel() * trim_ratio)
        if k == 0:
            var = torch.var(values, unbiased=False)
            return float(torch.clamp(var, min=eps).item())
        sorted_vals, _ = torch.sort(values)
        kept = sorted_vals[k:-k] if (2 * k) < values.numel() else sorted_vals
        if kept.numel() == 0:
            kept = sorted_vals
        var = torch.var(kept, unbiased=False)
        return float(torch.clamp(var, min=eps).item())

    @staticmethod
    def _global_variance_1d(values: torch.Tensor, eps: float) -> float:
        """Distributed variance of a 1-D tensor via first and second moments.

        Uses two all-reduce sums so that the estimate is consistent with a
        single-process computation over the concatenated batch.
        """
        if values.numel() == 0:
            return 0.0
        local_count = torch.tensor(
            [values.numel()], device=values.device, dtype=torch.float64
        )
        local_sum = torch.sum(values.to(torch.float64))
        local_sqsum = torch.sum((values.to(torch.float64)) ** 2)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_count)
            dist.all_reduce(local_sum)
            dist.all_reduce(local_sqsum)
        count = local_count.item()
        mean = local_sum / max(count, 1.0)
        mean_sq = local_sqsum / max(count, 1.0)
        var = torch.clamp(mean_sq - mean * mean, min=eps)
        return float(var.item())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_stats(
        self,
        *,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        expert_mask: torch.Tensor,
        current_step: int,
    ) -> None:
        """Refresh the three EMA proxies from a fresh mini-batch.

        This routine is a no-op when called more frequently than
        ``mu_update_freq``; the throttling is used both to reduce
        micro-batch noise and to keep the controller's runtime cost under
        one percent of the training-loop wall time.

        Parameters
        ----------
        logprob : torch.Tensor
            Token-level log-probabilities of shape ``[B, T]``.
        action_mask : torch.Tensor
            Boolean or float mask of shape ``[B, T]`` selecting response
            (loss-bearing) tokens.
        advantages : torch.Tensor
            Per-token advantages of shape ``[B, T]``; used only for
            sequences flagged as RL by ``expert_mask``.
        expert_mask : torch.Tensor
            Boolean mask of shape ``[B]``. ``True`` marks SFT (expert)
            sequences; ``False`` marks on-policy RL sequences.
        current_step : int
            Global optimiser step, consulted against ``mu_update_freq`` to
            decide whether an update is due.
        """
        if self._last_update_step is not None and (
            current_step - self._last_update_step < self.mu_update_freq
        ):
            return

        with torch.no_grad():
            rl_sel = ~expert_mask
            sft_sel = expert_mask

            # sigma_r^2: dispersion of sequence-level normalised advantages.
            if rl_sel.any():
                adv_rl = advantages[rl_sel]
                act_mask_rl = action_mask[rl_sel]
                seq_adv = masked_mean(adv_rl, act_mask_rl, axis=-1)
                sigma_r2 = self._global_variance_1d(seq_adv.detach(), self.eps)
            else:
                sigma_r2 = 0.0

            # sigma_s^2: trimmed variance of length-normalised per-sample NLL.
            if sft_sel.any():
                logprob_sft = logprob[sft_sel]
                act_mask_sft = action_mask[sft_sel]
                nll = -logprob_sft
                nll_seq = masked_mean(nll, act_mask_sft, axis=-1)
                sigma_s2 = self._trimmed_variance(
                    nll_seq.detach(), self.trim_ratio_sft, self.eps
                )
            else:
                sigma_s2 = 0.0

            # Delta_g^2: coefficient-space disagreement on response tokens.
            token_prob = torch.exp(logprob)
            if self.use_phi_in_sft:
                g_s_token = -(token_prob * (1 - token_prob)).detach()
            else:
                g_s_token = -torch.ones_like(logprob)

            g_r_token = torch.zeros_like(logprob)
            if rl_sel.any():
                g_r_token[rl_sel] = -advantages[rl_sel]

            delta_token = g_s_token - g_r_token
            var_seq = masked_mean(delta_token * delta_token, action_mask, axis=-1)
            delta_g2 = torch.mean(var_seq).item()

            # EMA update.
            beta = self.ema_beta
            self.ema_sigma_r2 = (
                sigma_r2
                if self.ema_sigma_r2 is None
                else beta * self.ema_sigma_r2 + (1 - beta) * sigma_r2
            )
            self.ema_sigma_s2 = (
                sigma_s2
                if self.ema_sigma_s2 is None
                else beta * self.ema_sigma_s2 + (1 - beta) * sigma_s2
            )
            self.ema_delta_g2 = (
                delta_g2
                if self.ema_delta_g2 is None
                else beta * self.ema_delta_g2 + (1 - beta) * delta_g2
            )

            self._last_update_step = current_step

    def compute_mu(
        self,
        *,
        alpha: Optional[float] = None,
        mu_prior: Optional[float] = None,
        blend_weight: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Return the guarded mixing weight :math:`\\mu` for the next step.

        The pipeline is: closed-form estimate :math:`\\to` EMA smoothing
        :math:`\\to` optional cosine-prior blend :math:`\\to` per-step change
        cap :math:`\\to` hard clipping to :math:`[\\mu_\\min, \\mu_\\max]`.

        Parameters
        ----------
        alpha : float, optional
            Target ratio :math:`\\alpha_\\mathrm{ctrl}` from the outer
            KL-driven controller. Defaults to ``alpha_static``.
        mu_prior : float, optional
            Value of the schedule prior at the current step. If ``None``,
            the prior blend is disabled and the adaptive estimate is used
            directly.
        blend_weight : float, default 1.0
            Weight :math:`\\lambda` on the adaptive estimate in the prior
            blend; :math:`\\lambda = 0` recovers a purely prior-driven
            schedule, :math:`\\lambda = 1` follows the adaptive estimate.

        Returns
        -------
        mu : float
            The mixing weight to apply this step.
        stats : dict
            Diagnostic scalars suitable for TensorBoard / W&B logging.
        """
        alpha_eff = self.alpha_static if alpha is None else float(alpha)

        # Fallbacks when statistics are not yet ready.
        delta_g2 = self.ema_delta_g2 if self.ema_delta_g2 is not None else 0.0
        sigma_s2 = self.ema_sigma_s2 if self.ema_sigma_s2 is not None else 1.0
        sigma_r2 = self.ema_sigma_r2 if self.ema_sigma_r2 is not None else 1.0

        delta_g2_capped = float(delta_g2)
        if self.delta_g2_cap_ratio and self.delta_g2_cap_ratio > 0.0:
            cap_val = self.delta_g2_cap_ratio * (sigma_s2 + sigma_r2 + self.eps)
            if delta_g2_capped > cap_val:
                delta_g2_capped = cap_val

        denom = delta_g2_capped + sigma_s2 + sigma_r2 + self.eps
        mu_raw = (alpha_eff * delta_g2_capped + sigma_r2) / denom

        # EMA-smoothed adaptive candidate.
        mu_adaptive = self.ema_beta * self.mu + (1 - self.ema_beta) * mu_raw
        # These values are scalar controller state.  Clamp in Python rather
        # than through a default float32 tensor so an exact boundary such as
        # ``mu_max=0.8`` cannot become 0.8000000119 and violate strict callers.
        mu_adaptive = min(max(float(mu_adaptive), self.mu_min), self.mu_max)

        # Cosine-prior blend (guided).
        blend = float(max(0.0, min(1.0, blend_weight)))
        if mu_prior is not None:
            mu_candidate = (1.0 - blend) * float(mu_prior) + blend * mu_adaptive
        else:
            mu_candidate = mu_adaptive

        # Per-step change cap.
        delta = mu_candidate - self.mu
        if abs(delta) > self.mu_change_cap:
            mu_candidate = self.mu + (
                self.mu_change_cap if delta > 0 else -self.mu_change_cap
            )
        mu_candidate = min(max(float(mu_candidate), self.mu_min), self.mu_max)
        self.mu = mu_candidate

        stats = {
            "mu/raw": float(mu_raw),
            "mu/value": float(self.mu),
            "mu/sigma_s2": float(sigma_s2),
            "mu/sigma_r2": float(sigma_r2),
            "mu/delta_g2": float(delta_g2),
            "mu/delta_g2_capped": float(delta_g2_capped),
            "mu/alpha": float(alpha_eff),
            "mu/prior": float(mu_prior) if mu_prior is not None else float("nan"),
            "mu/blend_weight": blend,
        }
        return self.mu, stats


def mu_schedule(
    global_step: int,
    warmup_steps: int,
    decay_steps: int,
    mu_peak: float,
    mu_valley: float,
) -> float:
    """Warmup + cosine-decay schedule prior for :math:`\\mu`.

    The schedule ramps linearly from ``0`` to ``mu_peak`` over
    ``warmup_steps``, then decays with a cosine to ``mu_valley`` over the
    following ``decay_steps`` steps, and remains at ``mu_valley`` thereafter.
    This is the default prior consumed by :meth:`AdaptiveMuController.compute_mu`
    in the paper's experiments.
    """
    import math

    if global_step < warmup_steps:
        return (global_step / max(1, warmup_steps)) * mu_peak
    if global_step >= (warmup_steps + decay_steps):
        return mu_valley
    progress = (global_step - warmup_steps) / max(1, decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return mu_valley + (mu_peak - mu_valley) * cosine
