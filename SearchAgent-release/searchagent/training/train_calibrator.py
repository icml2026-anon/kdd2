from __future__ import annotations

import argparse
import copy
import logging
import math
import os
from typing import List

import numpy as np
import torch

from searchagent.agent.calibration import Calibrator
from searchagent.agent.llm_backbone import LLMBackbone
from searchagent.config import CalibrationConfig, LLMConfig, SearchAgentConfig
from searchagent.environment.serp import OrcasSERPLoader, TRECSERPLoader, YandexSERPLoader
from searchagent.training import losses as L
from searchagent.training.data_builder import MOMENT_NAMES, PriorCache, build_dataset
from searchagent.utils.logging import setup_logging


logger = logging.getLogger(__name__)
DATASET_PATHS = {
    "trec2012": "data/search_logs/trec_session2012_fixed.xml",
    "trec2013": "data/search_logs/trec_session2013.xml",
    "trec2014": "data/search_logs/trec_session2014.xml",
    "yandex": "data/yandex/train.gz",
    "orcas": "data/search_logs/orcas.tsv.gz",
}


def _load(name: str, max_sessions: int = None):
    path = DATASET_PATHS[name]
    if name == "yandex":
        return YandexSERPLoader(path, max_sessions=max_sessions or 200).sessions
    if name == "orcas":
        return OrcasSERPLoader(path, max_sessions=max_sessions or 200).sessions
    return TRECSERPLoader(path).sessions


def _to_tensors(actions):
    return (
        torch.tensor(np.stack([a.feats for a in actions]), dtype=torch.float32),
        torch.tensor(np.stack([a.gate_feats for a in actions]), dtype=torch.float32),
        torch.tensor(np.stack([a.log_p_llm for a in actions]), dtype=torch.float32),
        torch.tensor([a.label for a in actions], dtype=torch.long),
    )


def _click_tensors(clicks):
    return [
        (
            torch.tensor(c.exam_feats, dtype=torch.float32),
            torch.tensor(c.attr_feats, dtype=torch.float32),
            torch.tensor(c.ranks, dtype=torch.float32),
            c.clicked_idx,
        )
        for c in clicks
    ]


def _build_split(names, cache, max_sessions):
    actions, clicks, moments = [], [], []
    for name in names:
        a, c, m = build_dataset(_load(name, max_sessions), name, llm=None, prior_cache=cache, max_sessions=max_sessions)
        actions.extend(a)
        clicks.extend(c)
        moments.extend(m)
    return actions, clicks, moments


def build_cache(args):
    setup_logging("INFO")
    llm = LLMBackbone(LLMConfig())
    cache = PriorCache(os.path.join(args.out, "prior_cache.json"))
    names = list(dict.fromkeys(args.train + args.val + args.test))
    for name in names:
        if name in DATASET_PATHS:
            build_dataset(_load(name, args.max_sessions), name, llm, cache, args.max_sessions)
    cache.save()


def _slice_action(tensors, indices):
    return tuple(t[indices] for t in tensors)


def _validation_objective(calibrator, actions, clicks) -> float:
    calibrator.eval()
    with torch.no_grad():
        values = []
        if actions:
            values.append(float(L.action_loss(calibrator, _to_tensors(actions))))
        if clicks:
            values.append(float(L.click_loss(calibrator, _click_tensors(clicks))))
    if not values:
        return float("inf")
    if len(values) == 1:
        return values[0]
    return 0.5 * values[0] + 0.5 * values[1]


def train(args):
    setup_logging("INFO")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.config:
        full_config = SearchAgentConfig.from_yaml(args.config)
        cfg = full_config.calibration
        n_intent = len(full_config.intent.states)
    else:
        cfg = CalibrationConfig()
        n_intent = 3
    epochs = args.epochs if args.epochs is not None else cfg.epochs
    lr = args.lr if args.lr is not None else cfg.lr
    cache = PriorCache(os.path.join(args.out, "prior_cache.json"))
    if not cache.store:
        raise SystemExit("Prior cache empty. Run build-cache first.")
    train_actions, train_clicks, _ = _build_split(args.train, cache, args.max_sessions)
    val_names = args.val if args.val else args.test
    val_actions, val_clicks, _ = _build_split(val_names, cache, args.max_sessions)
    if not train_actions:
        raise SystemExit("No training action examples.")
    calibrator = Calibrator(
        n_intent=n_intent,
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        dropout=cfg.dropout,
        alpha_max=cfg.alpha_max,
        exam_dim=cfg.exam_dim,
        attr_dim=cfg.attr_dim,
        epsilon=cfg.epsilon,
    )
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    action_tensors = _to_tensors(train_actions)
    click_tensors = _click_tensors(train_clicks)
    best_state = copy.deepcopy(calibrator.state_dict())
    best_val = float("inf")
    stale = 0
    rng = np.random.RandomState(args.seed)
    for epoch in range(epochs):
        calibrator.train()
        action_order = rng.permutation(len(train_actions))
        click_order = rng.permutation(len(train_clicks)) if train_clicks else np.asarray([], dtype=int)
        n_action_batches = max(1, math.ceil(len(train_actions) / cfg.action_batch_size))
        n_click_batches = max(1, math.ceil(len(train_clicks) / cfg.click_batch_size)) if train_clicks else 1
        steps = max(n_action_batches, n_click_batches)
        epoch_losses = []
        for batch_index in range(steps):
            a_start = (batch_index % n_action_batches) * cfg.action_batch_size
            a_idx = action_order[a_start:a_start + cfg.action_batch_size]
            if len(a_idx) == 0:
                a_idx = action_order[: min(cfg.action_batch_size, len(action_order))]
            action_batch = _slice_action(action_tensors, a_idx)
            if train_clicks:
                c_start = (batch_index % n_click_batches) * cfg.click_batch_size
                c_idx = click_order[c_start:c_start + cfg.click_batch_size]
                if len(c_idx) == 0:
                    c_idx = click_order[: min(cfg.click_batch_size, len(click_order))]
                click_batch = [click_tensors[i] for i in c_idx]
            else:
                click_batch = []
            optimizer.zero_grad()
            loss, _ = L.total_offline_loss(calibrator, action_batch, click_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        val = _validation_objective(calibrator, val_actions, val_clicks)
        logger.info("epoch=%d train_nll=%.6f val_nll=%.6f", epoch + 1, float(np.mean(epoch_losses)), val)
        if best_val - val >= cfg.early_stop_min_delta:
            best_val = val
            best_state = copy.deepcopy(calibrator.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= cfg.early_stop_patience:
            break
    calibrator.load_state_dict(best_state)
    os.makedirs(args.out, exist_ok=True)
    checkpoint = os.path.join(args.out, "calibrator.pt")
    torch.save({
        "state_dict": calibrator.state_dict(),
        "moment_names": MOMENT_NAMES,
        "best_validation_nll": best_val,
        "dims": {
            "n_intent": n_intent,
            "hidden1": cfg.hidden1,
            "hidden2": cfg.hidden2,
            "dropout": cfg.dropout,
            "alpha_max": cfg.alpha_max,
            "exam_dim": cfg.exam_dim,
            "attr_dim": cfg.attr_dim,
            "epsilon": cfg.epsilon,
        },
    }, checkpoint)


def _action_acc(calibrator, actions):
    if not actions:
        return float("nan")
    feats, gate, prior, labels = _to_tensors(actions)
    pred = calibrator.posterior.log_prob(feats, prior, gate).argmax(dim=-1)
    return float((pred == labels).float().mean())


def _click_acc(calibrator, clicks):
    if not clicks:
        return float("nan")
    correct = 0
    for c in clicks:
        exam = torch.tensor(c.exam_feats, dtype=torch.float32)
        attr = torch.tensor(c.attr_feats, dtype=torch.float32)
        ranks = torch.tensor(c.ranks, dtype=torch.float32)
        correct += int(calibrator.click.log_prob(exam, attr, ranks).argmax().item() == c.clicked_idx)
    return correct / len(clicks)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    cache_parser = sub.add_parser("build-cache")
    cache_parser.add_argument("--train", nargs="+", default=["trec2012", "trec2013"])
    cache_parser.add_argument("--val", nargs="*", default=["trec2014"])
    cache_parser.add_argument("--test", nargs="*", default=[])
    cache_parser.add_argument("--out", default="output/calib")
    cache_parser.add_argument("--max_sessions", type=int, default=None)
    cache_parser.set_defaults(func=build_cache)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--train", nargs="+", default=["trec2012", "trec2013"])
    train_parser.add_argument("--val", nargs="*", default=["trec2014"])
    train_parser.add_argument("--test", nargs="*", default=[])
    train_parser.add_argument("--out", default="output/calib")
    train_parser.add_argument("--max_sessions", type=int, default=None)
    train_parser.add_argument("--config", type=str, default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
