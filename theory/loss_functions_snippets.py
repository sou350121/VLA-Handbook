"""
Minimal PyTorch snippets for common VLA / VTLA loss functions.

This file is meant for copy-paste into training codebases. It intentionally avoids
framework-specific dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def mse_bc_loss(pred_action: torch.Tensor, gt_action: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Behavior cloning MSE for continuous actions.

    pred_action: [B, T, D]
    gt_action:   [B, T, D]
    mask:        [B, T] or [B, T, 1] (1 keeps, 0 masks out)
    """
    diff = pred_action - gt_action
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        diff = diff * mask
        denom = mask.sum().clamp_min(1.0)
        return (diff.pow(2).sum() / denom)
    return diff.pow(2).mean()


def huber_bc_loss(pred_action: torch.Tensor, gt_action: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Huber / SmoothL1."""
    return F.smooth_l1_loss(pred_action, gt_action, beta=beta)


def gaussian_nll_loss(mu: torch.Tensor, log_var: torch.Tensor, gt: torch.Tensor, clamp_min: float = -10.0, clamp_max: float = 5.0) -> torch.Tensor:
    """
    Heteroscedastic Gaussian NLL (diagonal covariance).

    mu:      [B, T, D]
    log_var: [B, T, D]
    gt:      [B, T, D]
    """
    log_var = log_var.clamp(clamp_min, clamp_max)
    var = torch.exp(log_var).clamp_min(1e-6)
    return 0.5 * (((gt - mu) ** 2) / var + log_var).mean()


def token_ce_loss(logits: torch.Tensor, target_bins: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Cross entropy for discretized actions.

    logits: [B, T, D, N]
    target: [B, T, D]
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_bins.reshape(-1),
        ignore_index=ignore_index,
    )


@dataclass
class GMMDiag:
    """Diagonal-covariance GMM parameters."""

    logits_pi: torch.Tensor  # [B, T, K]
    mu: torch.Tensor         # [B, T, K, D]
    log_std: torch.Tensor    # [B, T, K, D]


def gmm_diag_nll(gmm: GMMDiag, gt: torch.Tensor, clamp_log_std: Tuple[float, float] = (-7.0, 3.0)) -> torch.Tensor:
    """
    Negative log-likelihood for diagonal GMM.

    gt: [B, T, D]
    """
    logits_pi, mu, log_std = gmm.logits_pi, gmm.mu, gmm.log_std.clamp(*clamp_log_std)
    B, T, K = logits_pi.shape
    D = gt.size(-1)

    # [B, T, 1, D] - [B, T, K, D] -> [B, T, K, D]
    diff = gt.unsqueeze(-2) - mu
    inv_var = torch.exp(-2.0 * log_std)

    # log N(x; mu, diag(var))
    log_det = 2.0 * log_std.sum(dim=-1)  # [B, T, K]
    quad = (diff.pow(2) * inv_var).sum(dim=-1)  # [B, T, K]
    log_norm = D * math.log(2.0 * math.pi)
    log_prob = -0.5 * (log_norm + log_det + quad)  # [B, T, K]

    log_pi = F.log_softmax(logits_pi, dim=-1)  # [B, T, K]
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # [B, T]
    return (-log_mix).mean()


def action_smoothness_loss(action: torch.Tensor, lambda_v: float = 1.0, lambda_a: float = 0.0) -> torch.Tensor:
    """
    Smoothness regularization on action sequences.

    action: [B, T, D]
    """
    v = action[:, 1:] - action[:, :-1]
    loss = lambda_v * (v.pow(2).mean())
    if lambda_a > 0.0 and action.size(1) >= 3:
        a = v[:, 1:] - v[:, :-1]
        loss = loss + lambda_a * (a.pow(2).mean())
    return loss


def info_nce_loss(q: torch.Tensor, k: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE for batch-aligned positives (CLIP-style).

    q: [B, D]  (e.g., image embeddings)
    k: [B, D]  (e.g., text embeddings)
    """
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = (q @ k.t()) / tau  # [B, B]
    labels = torch.arange(q.size(0), device=q.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


