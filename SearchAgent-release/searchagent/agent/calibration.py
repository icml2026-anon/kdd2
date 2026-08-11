from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ABLATE = None
CALIB_ACTIONS = [
    "click_result",
    "skip",
    "next_page",
    "reformulate",
    "new_query",
    "stop_satisfied",
    "stop_unsatisfied",
]
ACTION_INDEX = {a: i for i, a in enumerate(CALIB_ACTIONS)}
N_ACTIONS = len(CALIB_ACTIONS)
EPSILON = 1e-8


def _scaled(value: str, mapping: Dict[str, float], default: float = 0.5) -> float:
    return float(mapping.get(value, default))


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPSILON, None)
    p = p / p.sum()
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


@dataclass
class StateFeatures:
    persona: List[float]
    cognitive: List[float]
    intent: List[float]
    memory: List[float]
    serp_quality: float
    llm_entropy: float

    def vector(self) -> np.ndarray:
        return np.asarray(self.persona + self.cognitive + self.intent + self.memory + [self.serp_quality, self.llm_entropy], dtype=np.float32)

    @staticmethod
    def dim(n_intent: int = 3) -> int:
        return 3 + 4 + n_intent + 5 + 2


class StateFeaturizer:
    def __init__(self, n_intent: int = 3):
        self.n_intent = n_intent

    def build(self, persona, cognitive, intent_dist: List[float], memory_feats: Dict[str, float], serp_quality: float, llm_probs: np.ndarray) -> StateFeatures:
        persona_vec = [
            float(getattr(persona, "z_exp", _scaled(getattr(persona, "search_expertise", "intermediate"), {"novice": 0.0, "intermediate": 0.5, "expert": 1.0}))),
            float(getattr(persona, "z_style", _scaled(getattr(persona, "decision_style", "deliberate"), {"impulsive": 0.0, "deliberate": 1.0, "price_sensitive": 0.5}))),
            float(getattr(persona, "z_pat", _scaled(getattr(persona, "patience_level", "medium"), {"low": 0.0, "medium": 0.5, "high": 1.0}))),
        ]
        cognitive_vec = [
            float(np.clip(getattr(cognitive, "need_clarity", 0.3), 0.0, 1.0)),
            float(np.clip(getattr(cognitive, "knowledge_state", 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cognitive, "frustration", 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cognitive, "exploration_focus", 0.3), 0.0, 1.0)),
        ]
        intent = list(np.asarray(intent_dist, dtype=np.float64)[: self.n_intent])
        while len(intent) < self.n_intent:
            intent.append(0.0)
        intent = [float(x) for x in intent]
        memory = [
            float(np.clip(memory_feats.get("n_click", 0.0), 0.0, 1.0)),
            float(np.clip(memory_feats.get("n_skip", 0.0), 0.0, 1.0)),
            float(np.clip(memory_feats.get("last_satisfaction", 0.0), 0.0, 1.0)),
            float(np.clip(memory_feats.get("page", 0.0), 0.0, 1.0)),
            float(np.clip(memory_feats.get("step", 0.0), 0.0, 1.0)),
        ]
        return StateFeatures(persona_vec, cognitive_vec, intent, memory, float(np.clip(serp_quality, 0.0, 1.0)), _entropy(llm_probs))


class BehaviorCalibrationNet(nn.Module):
    def __init__(self, feat_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.1, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_actions),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class TrustGate(nn.Module):
    def __init__(self, alpha_max: float = 2.0):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.beta0 = nn.Parameter(torch.zeros(()))
        self.beta_h = nn.Parameter(torch.zeros(()))
        self.beta_q = nn.Parameter(torch.zeros(()))
        self.beta_free = nn.Parameter(torch.zeros(4))

    def forward(self, gate_feats: torch.Tensor) -> torch.Tensor:
        h = gate_feats[..., 0]
        q = gate_feats[..., 1]
        free = gate_feats[..., 2:6]
        z = self.beta0 - F.softplus(self.beta_h) * h + F.softplus(self.beta_q) * q + (free * self.beta_free).sum(dim=-1)
        return self.alpha_max * torch.sigmoid(z).unsqueeze(-1)


class ActionPosterior(nn.Module):
    def __init__(self, feat_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.1, alpha_max: float = 2.0, epsilon: float = EPSILON):
        super().__init__()
        self.f = BehaviorCalibrationNet(feat_dim, hidden1, hidden2, dropout)
        self.gate = TrustGate(alpha_max)
        self.epsilon = float(epsilon)

    def residual(self, feats: torch.Tensor, log_p_llm: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(log_p_llm, dim=-1)
        ell = log_p_llm
        ell_bar = ell - (p * ell).sum(dim=-1, keepdim=True)
        raw = self.f(feats)
        f_bar = raw - (p * raw).sum(dim=-1, keepdim=True)
        numerator = (p * f_bar * ell_bar).sum(dim=-1, keepdim=True)
        denominator = (p * ell_bar.square()).sum(dim=-1, keepdim=True) + self.epsilon
        gamma = numerator / denominator
        return f_bar - gamma * ell_bar

    def forward(self, feats: torch.Tensor, log_p_llm: torch.Tensor, gate_feats: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(gate_feats)
        if ABLATE == "no_gate":
            alpha = torch.ones_like(alpha)
        residual = self.residual(feats, log_p_llm)
        if ABLATE == "no_f":
            residual = torch.zeros_like(residual)
        return alpha * log_p_llm + residual

    def log_prob(self, feats: torch.Tensor, log_p_llm: torch.Tensor, gate_feats: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward(feats, log_p_llm, gate_feats), dim=-1)


class ClickModel(nn.Module):
    def __init__(self, exam_dim: int = 5, attr_dim: int = 3):
        super().__init__()
        self.exam = nn.Sequential(nn.Linear(exam_dim, 16), nn.GELU(), nn.Linear(16, 1))
        self.attr = nn.Sequential(
            nn.Linear(attr_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, exam_context: torch.Tensor, attr_feats: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
        if exam_context.ndim == 1:
            exam_context = exam_context.unsqueeze(0)
        lam = F.softplus(self.exam(exam_context)).reshape(-1)
        if lam.numel() != 1:
            raise ValueError("click selector expects one examination context per SERP")
        psi = self.attr(attr_feats).squeeze(-1)
        ranks = torch.clamp(ranks.to(dtype=psi.dtype, device=psi.device), min=1.0)
        return psi - lam[0] * torch.log(ranks)

    def log_prob(self, exam_context: torch.Tensor, attr_feats: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward(exam_context, attr_feats, ranks), dim=-1)


class Calibrator(nn.Module):
    def __init__(self, n_intent: int = 3, hidden: Optional[int] = None, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.1, alpha_max: float = 2.0, exam_dim: int = 5, attr_dim: int = 3, epsilon: float = EPSILON):
        super().__init__()
        if hidden is not None:
            hidden1 = int(hidden)
        self.n_intent = n_intent
        self.feat_dim = StateFeatures.dim(n_intent)
        self.posterior = ActionPosterior(self.feat_dim, hidden1, hidden2, dropout, alpha_max, epsilon)
        self.click = ClickModel(exam_dim, attr_dim)
        self.featurizer = StateFeaturizer(n_intent)

    @torch.no_grad()
    def action_probs(self, feats_vec: np.ndarray, log_p_llm: np.ndarray, gate_feats: np.ndarray) -> np.ndarray:
        feats = torch.as_tensor(feats_vec, dtype=torch.float32).unsqueeze(0)
        log_prior = torch.as_tensor(log_p_llm, dtype=torch.float32).unsqueeze(0)
        gate = torch.as_tensor(gate_feats, dtype=torch.float32).unsqueeze(0)
        return self.posterior.log_prob(feats, log_prior, gate).exp().squeeze(0).cpu().numpy()

    @torch.no_grad()
    def click_probs(self, exam_context: np.ndarray, attr_feats: np.ndarray, ranks: np.ndarray) -> np.ndarray:
        if ABLATE == "no_click":
            ranks = np.maximum(np.asarray(ranks, dtype=np.float64), 1.0)
            p = 1.0 / ranks
            return (p / p.sum()).astype(np.float32)
        exam = torch.as_tensor(exam_context, dtype=torch.float32)
        attr = torch.as_tensor(attr_feats, dtype=torch.float32)
        rank_t = torch.as_tensor(ranks, dtype=torch.float32)
        return self.click.log_prob(exam, attr, rank_t).exp().cpu().numpy()
