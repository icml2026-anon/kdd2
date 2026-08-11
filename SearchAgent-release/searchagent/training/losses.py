from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def action_loss(calibrator, batch) -> torch.Tensor:
    feats, gate, log_p, labels = batch
    return F.nll_loss(calibrator.posterior.log_prob(feats, log_p, gate), labels)


def click_loss(calibrator, click_batch: List) -> torch.Tensor:
    if not click_batch:
        return next(calibrator.parameters()).sum() * 0.0
    values = []
    for item in click_batch:
        if len(item) == 4:
            exam, attr, ranks, clicked = item
        else:
            exam, attr, clicked = item
            ranks = torch.arange(1, attr.shape[0] + 1, dtype=attr.dtype, device=attr.device)
        values.append(-calibrator.click.log_prob(exam, attr, ranks)[clicked])
    return torch.stack(values).mean()


def total_offline_loss(calibrator, action_batch, click_batch, **kwargs):
    la = action_loss(calibrator, action_batch)
    lc = click_loss(calibrator, click_batch)
    total = 0.5 * la + 0.5 * lc
    return total, {"L_act": float(la.detach()), "L_click": float(lc.detach()), "total": float(total.detach())}
