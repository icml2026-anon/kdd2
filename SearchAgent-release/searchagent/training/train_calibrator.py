"""Train the shared SearchAgent++ calibrator.

Usage:
  # Step 1 (one-time, uses API to fill the prior cache, resumable):
  python -m searchagent.training.train_calibrator build-cache \
      --train trec2012 trec2013 --out output/calib

  # Step 2 (offline, no API): Phase A imitation training
  python -m searchagent.training.train_calibrator train \
      --train trec2012 trec2013 --test trec2014 --out output/calib --epochs 200
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List

import numpy as np
import torch

from searchagent.agent.calibration import Calibrator
from searchagent.agent.llm_backbone import LLMBackbone
from searchagent.config import LLMConfig
from searchagent.environment.serp import TRECSERPLoader, YandexSERPLoader, OrcasSERPLoader
from searchagent.training.data_builder import (
    PriorCache, build_dataset, MOMENT_NAMES,
)
from searchagent.training import losses as L
from searchagent.utils.logging import setup_logging

logger = logging.getLogger(__name__)

DATASET_PATHS = {
    "trec2012": "data/search_logs/trec_session2012_fixed.xml",
    "trec2013": "data/search_logs/trec_session2013.xml",
    "trec2014": "data/search_logs/trec_session2014.xml",
    "yandex":   "data/yandex/train.gz",
    "orcas":    "/root/orcas.tsv.gz",
}


def _load(name: str, max_sessions: int = None):
    path = DATASET_PATHS[name]
    if name == "yandex":
        return YandexSERPLoader(path, max_sessions=max_sessions or 200).sessions
    if name == "orcas":
        return OrcasSERPLoader(path, max_sessions=max_sessions or 200).sessions
    return TRECSERPLoader(path).sessions


def _to_tensors(actions):
    feats = torch.tensor(np.stack([a.feats for a in actions]), dtype=torch.float32)
    gate = torch.tensor(np.stack([a.gate_feats for a in actions]), dtype=torch.float32)
    log_p = torch.tensor(np.stack([a.log_p_llm for a in actions]), dtype=torch.float32)
    labels = torch.tensor([a.label for a in actions], dtype=torch.long)
    return feats, gate, log_p, labels


def _click_tensors(clicks):
    out = []
    for c in clicks:
        out.append((torch.tensor(c.exam_feats), torch.tensor(c.attr_feats), c.clicked_idx))
    return out


def build_cache(args):
    setup_logging("INFO")
    llm = LLMBackbone(LLMConfig())
    cache = PriorCache(os.path.join(args.out, "prior_cache.json"))
    for name in args.train + args.test:
        if name not in DATASET_PATHS:
            continue
        logger.info("Building cache for %s ...", name)
        build_dataset(_load(name, args.max_sessions), name, llm=llm, prior_cache=cache,
                      max_sessions=args.max_sessions)
    cache.save()
    logger.info("Cache build complete: %d entries", len(cache.store))


def _build_split(names, cache, max_sessions):
    A, C, M = [], [], []
    for name in names:
        a, c, m = build_dataset(_load(name, max_sessions), name, llm=None,
                                prior_cache=cache, max_sessions=max_sessions)
        A += a; C += c; M += m
    return A, C, M


def train(args):
    setup_logging("INFO")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Merge: yaml calibration params as base, CLI args as override (None = not set)
    from searchagent.config import SearchAgentConfig, CalibrationConfig
    if args.config:
        full_cfg = SearchAgentConfig.from_yaml(args.config)
        calib = full_cfg.calibration
        n_intent = len(full_cfg.intent.states)
    else:
        calib = CalibrationConfig()
        n_intent = 3
    epochs = args.epochs if args.epochs is not None else calib.epochs
    lr = args.lr if args.lr is not None else calib.lr
    lambda_click = args.lambda_click if args.lambda_click is not None else calib.lambda_click
    lambda_kl = args.lambda_kl if args.lambda_kl is not None else calib.lambda_kl
    lambda_cal = args.lambda_cal if args.lambda_cal is not None else calib.lambda_cal
    logger.info("Training params: epochs=%d lr=%.4f λ_click=%.2f λ_kl=%.2f λ_cal=%.2f",
                epochs, lr, lambda_click, lambda_kl, lambda_cal)

    cache = PriorCache(os.path.join(args.out, "prior_cache.json"))
    if not cache.store:
        raise SystemExit("Prior cache empty. Run `build-cache` first.")

    tr_a, tr_c, tr_m = _build_split(args.train, cache, args.max_sessions)
    te_a, te_c, te_m = _build_split(args.test, cache, args.max_sessions)
    logger.info("Train: %d actions / %d clicks / %d sessions", len(tr_a), len(tr_c), len(tr_m))
    logger.info("Test : %d actions / %d clicks / %d sessions", len(te_a), len(te_c), len(te_m))
    if not tr_a:
        raise SystemExit("No training actions (cache miss?). Re-run build-cache.")

    calibrator = Calibrator(n_intent=n_intent,
                            hidden=calib.hidden,
                            exam_dim=calib.exam_dim,
                            attr_dim=calib.attr_dim)
    opt = torch.optim.AdamW(calibrator.parameters(), lr=lr, weight_decay=1e-4)

    tr_batch = _to_tensors(tr_a)
    tr_clicks = _click_tensors(tr_c)
    real_phi = torch.tensor(np.stack([m.phi for m in tr_m]), dtype=torch.float32)

    for ep in range(epochs):
        calibrator.train()
        opt.zero_grad()
        loss, parts = L.total_offline_loss(
            calibrator, tr_batch, tr_clicks,
            lambda_click=lambda_click, lambda_kl=lambda_kl,
            lambda_cal=lambda_cal)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 5.0)
        opt.step()

        if (ep + 1) % args.eval_every == 0 or ep == 0:
            calibrator.eval()
            with torch.no_grad():
                te_acc = _action_acc(calibrator, te_a) if te_a else float("nan")
                tr_acc = _action_acc(calibrator, tr_a)
                te_clk = _click_acc(calibrator, te_c) if te_c else float("nan")
            logger.info("ep %4d  total=%.4f L_act=%.4f L_click=%.4f L_KL=%.4f "
                        "| train_acc=%.3f test_acc=%.3f test_click_acc=%.3f",
                        ep + 1, parts["total"], parts["L_act"], parts["L_click"],
                        parts["L_KL"], tr_acc, te_acc, te_clk)

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "calibrator.pt")
    torch.save({"state_dict": calibrator.state_dict(),
                "moment_names": MOMENT_NAMES,
                "dims": {"n_intent": n_intent, "hidden": calib.hidden,
                         "exam_dim": calib.exam_dim, "attr_dim": calib.attr_dim}}, ckpt)
    logger.info("Saved calibrator to %s (hidden=%d, n_intent=%d)",
                ckpt, calib.hidden, n_intent)


def _action_acc(calib, actions):
    if not actions:
        return float("nan")
    f, g, lp, y = _to_tensors(actions)
    pred = calib.posterior.log_prob(f, lp, g).argmax(dim=-1)
    return (pred == y).float().mean().item()


def _click_acc(calib, clicks):
    if not clicks:
        return float("nan")
    correct = 0
    for c in clicks:
        e = torch.tensor(c.exam_feats); a = torch.tensor(c.attr_feats)
        if calib.click.log_prob(e, a).argmax().item() == c.clicked_idx:
            correct += 1
    return correct / len(clicks)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("build-cache")
    pc.add_argument("--train", nargs="+", default=["trec2012", "trec2013"])
    pc.add_argument("--test", nargs="+", default=["trec2014"])
    pc.add_argument("--out", default="output/calib")
    pc.add_argument("--max_sessions", type=int, default=None)
    pc.set_defaults(func=build_cache)

    pt = sub.add_parser("train")
    pt.add_argument("--train", nargs="+", default=["trec2012", "trec2013"])
    pt.add_argument("--test", nargs="+", default=["trec2014"])
    pt.add_argument("--out", default="output/calib")
    pt.add_argument("--max_sessions", type=int, default=None)
    pt.add_argument("--config", type=str, default=None,
                    help="yaml config to read calibration params from (overrides defaults below)")
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--lr", type=float, default=None)
    pt.add_argument("--lambda_click", type=float, default=None)
    pt.add_argument("--lambda_kl", type=float, default=None)
    pt.add_argument("--lambda_cal", type=float, default=None)
    pt.add_argument("--eval_every", type=int, default=25)
    pt.add_argument("--seed", type=int, default=42)
    pt.set_defaults(func=train)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
