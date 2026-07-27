"""Training losses for SearchAgent++ calibration.

L(theta) = L_act + lambda_click * L_click + lambda_KL * L_KL + lambda_cal * L_cal

  L_act   : action imitation  (cross-entropy of P_theta(a*) on real steps)
  L_click : click imitation   (cross-entropy of P_theta(j*) on real clicks)
  L_KL    : KL(P_theta || p_LLM)  -- keep semantic prior, don't fully override
  L_cal   : calibration / entropy regularizer on trust gate alpha
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def action_loss(calibrator, batch) -> torch.Tensor:
    feats, gate, log_p, labels = batch
    logp = calibrator.posterior.log_prob(feats, log_p, gate)   # [B,A]
    return F.nll_loss(logp, labels)


def kl_to_prior(calibrator, batch) -> torch.Tensor:
    """KL(P_theta || p_LLM) averaged over the action batch."""
    feats, gate, log_p, _ = batch
    logp_theta = calibrator.posterior.log_prob(feats, log_p, gate)
    p_theta = logp_theta.exp()
    # log_p is log p_LLM (already normalized in CALIB order)
    return (p_theta * (logp_theta - log_p)).sum(dim=-1).mean()


def click_loss(calibrator, click_batch: List) -> torch.Tensor:
    """Sum CE over variable-length click examples (one real click each)."""
    if not click_batch:
        return torch.zeros((), requires_grad=True)
    losses = []
    for exam, attr, clicked in click_batch:
        logp = calibrator.click.log_prob(exam, attr)           # [n_results]
        losses.append(-logp[clicked])
    return torch.stack(losses).mean()


def moment_loss(sim_phi: torch.Tensor, real_phi: torch.Tensor) -> torch.Tensor:
    """|| mean simulated moments - mean real moments ||^2."""
    return F.mse_loss(sim_phi.mean(dim=0), real_phi.mean(dim=0))


def trust_reg(calibrator, batch) -> torch.Tensor:
    """Keep alpha away from collapse (regularize toward ~1.0)."""
    feats, gate, log_p, _ = batch
    alpha = calibrator.posterior.gate(gate)                    # [B,1]
    return ((alpha - 1.0) ** 2).mean()


def total_offline_loss(
    calibrator, action_batch, click_batch,
    lambda_click: float = 1.0, lambda_kl: float = 0.1, lambda_cal: float = 0.05,
):
    """Phase A: imitation + KL + trust reg (no rollouts needed)."""
    la = action_loss(calibrator, action_batch)
    lc = click_loss(calibrator, click_batch)
    lkl = kl_to_prior(calibrator, action_batch)
    lcal = trust_reg(calibrator, action_batch)
    total = la + lambda_click * lc + lambda_kl * lkl + lambda_cal * lcal
    return total, dict(L_act=la.item(), L_click=float(lc),
                       L_KL=lkl.item(), L_cal=lcal.item(), total=total.item())
