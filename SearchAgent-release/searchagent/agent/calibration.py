"""SearchAgent++ — LLM-prior Behavioral Posterior Calibration.

Replaces hand-weighted utilities with a calibration policy learned from real
search logs:

    P_theta(a | Omega_t)  proportional to  p_LLM(a | Omega_t)^{alpha_theta(Omega_t)}
                                            * exp(f_theta(a, Omega_t))

The LLM remains API-only (DeepSeek). Only the small calibration head defined
here is trained with torch. Three learned components:

  1. BehaviorCalibrationNet  f_theta(a, Omega_t)        -- learned action utility
  2. TrustGate               alpha_theta(Omega_t)        -- adaptive LLM trust
  3. ClickModel              examination * attractiveness -- learned click choice

Cognitive state (knowledge/frustration/focus) is updated by interpretable
hand rules in CognitiveSearchState (searchagent/agent/search_agent.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ablation switch (inference-time component disabling, no retraining).
# None = full model. Set to one of: "no_f" (drop BehaviorCalibrationNet),
# "no_gate" (force TrustGate alpha=1), "no_click" (ClickModel -> position bias).
ABLATE = None

# Macro action space (must match searchagent.agent.prompts.MACRO_ACTIONS ordering
# for the click-style action, plus the dynamics-only actions the simulator uses).
CALIB_ACTIONS = [
    "click_result", "skip", "next_page", "reformulate",
    "new_query", "stop_satisfied", "stop_unsatisfied",
]
ACTION_INDEX = {a: i for i, a in enumerate(CALIB_ACTIONS)}
N_ACTIONS = len(CALIB_ACTIONS)

# Categorical persona vocabularies (one-hot encoded in the featurizer).
EXPERTISE = ["novice", "intermediate", "expert"]
AD_SENS = ["ad_averse", "ad_neutral", "ad_tolerant"]
DEC_STYLE = ["impulsive", "deliberate", "price_sensitive"]
PATIENCE = ["low", "medium", "high"]


def _onehot(value: str, vocab: List[str]) -> List[float]:
    v = [0.0] * len(vocab)
    if value in vocab:
        v[vocab.index(value)] = 1.0
    return v


@dataclass
class StateFeatures:
    """Numeric view of Omega_t consumed by every calibration module."""

    persona: List[float]       # one-hot expertise+ad+style+patience + age scalar
    cognitive: List[float]     # [need_clarity, knowledge, frustration, focus]
    intent: List[float]        # soft distribution over intent states
    memory: List[float]        # [n_click, n_skip, ad_clicked, last_satis, page, step]
    serp_quality: float        # mean relevance / quality proxy in [0,1]
    llm_entropy: float         # H(p_LLM(a|Omega_t)) normalized to [0,1]
    dataset_stats: List[float] = None  # [clicks/session, abandon, reform] (dataset-aware)

    def vector(self) -> np.ndarray:
        return np.array(
            self.persona
            + self.cognitive
            + self.intent
            + self.memory
            + [self.serp_quality, self.llm_entropy],
            dtype=np.float32,
        )

    @staticmethod
    def dim(n_intent: int = 3) -> int:
        persona_dim = len(EXPERTISE) + len(AD_SENS) + len(DEC_STYLE) + len(PATIENCE) + 1
        return persona_dim + 4 + n_intent + 6 + 2


class StateFeaturizer:
    """Builds StateFeatures from agent-side primitives (no torch, no grad)."""

    def __init__(self, n_intent: int = 3):
        self.n_intent = n_intent

    def build(
        self,
        persona,
        cognitive,           # CognitiveSearchState
        intent_dist: List[float],
        memory_feats: Dict[str, float],
        serp_quality: float,
        llm_probs: np.ndarray,
        dataset_stats: List[float] = None,
    ) -> StateFeatures:
        persona_vec = (
            _onehot(getattr(persona, "search_expertise", "intermediate"), EXPERTISE)
            + _onehot(getattr(persona, "ad_sensitivity", "ad_neutral"), AD_SENS)
            + _onehot(getattr(persona, "decision_style", "deliberate"), DEC_STYLE)
            + _onehot(getattr(persona, "patience_level", "medium"), PATIENCE)
            + [min(max((getattr(persona, "age", 35) - 18) / 47.0, 0.0), 1.0)]
        )
        cog = [
            getattr(cognitive, "need_clarity", 0.3),
            getattr(cognitive, "knowledge_state", 0.0),
            getattr(cognitive, "frustration", 0.0),
            getattr(cognitive, "exploration_focus", 0.3),
        ]
        intent = list(intent_dist[: self.n_intent])
        while len(intent) < self.n_intent:
            intent.append(0.0)
        mem = [
            memory_feats.get("n_click", 0.0),
            memory_feats.get("n_skip", 0.0),
            memory_feats.get("ad_clicked", 0.0),
            memory_feats.get("last_satisfaction", 0.0),
            memory_feats.get("page", 1.0),
            memory_feats.get("step", 0.0),
        ]
        ent = _entropy(llm_probs)
        ds = list(dataset_stats) if dataset_stats is not None else [0.0, 0.0, 0.0]
        return StateFeatures(persona_vec, cog, intent, mem, float(serp_quality), ent, ds)


def _entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(p))) if len(p) > 1 else 0.0


# ----------------------------------------------------------------------------
# Component 1: BehaviorCalibrationNet  f_theta(a, Omega_t)
# ----------------------------------------------------------------------------
class BehaviorCalibrationNet(nn.Module):
    """Learned per-action utility. MLP over state features -> one logit per action.

    Replaces the hand-designed U_persona + U_css + U_intent + U_dynamics terms.
    """

    def __init__(self, feat_dim: int, hidden: int = 64, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, feat_dim] -> [B, n_actions]
        return self.net(feats)


# ----------------------------------------------------------------------------
# Component 2: TrustGate  alpha_theta(Omega_t) in (0, ~2)
# ----------------------------------------------------------------------------
class TrustGate(nn.Module):
    """Adaptive trust in the LLM prior.

    alpha = alpha_max * sigmoid(v . [H(p_LLM), step, n_click, frustration,
                                     knowledge, serp_quality])
    High LLM entropy / fatigue / poor SERP -> lower alpha -> trust behavior model more.
    """

    def __init__(self, alpha_max: float = 2.0):
        super().__init__()
        self.alpha_max = alpha_max
        self.lin = nn.Linear(6, 1)
        # Initialize toward alpha ~ 1.0 (sigmoid(0)=0.5 * alpha_max=2.0).
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, gate_feats: torch.Tensor) -> torch.Tensor:
        # gate_feats: [B, 6] -> alpha [B, 1]
        return self.alpha_max * torch.sigmoid(self.lin(gate_feats))


# ----------------------------------------------------------------------------
# Combined action posterior  P_theta(a | Omega_t)
# ----------------------------------------------------------------------------
class ActionPosterior(nn.Module):
    """P_theta(a) = softmax( alpha * log p_LLM(a) + f_theta(a, Omega) )."""

    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.f = BehaviorCalibrationNet(feat_dim, hidden)
        self.gate = TrustGate()

    def forward(
        self, feats: torch.Tensor, log_p_llm: torch.Tensor, gate_feats: torch.Tensor
    ) -> torch.Tensor:
        alpha = self.gate(gate_feats)                 # [B,1]
        if ABLATE == "no_gate":
            alpha = torch.ones_like(alpha)            # fixed trust, no adaptation
        f_term = self.f(feats)
        if ABLATE == "no_f":
            f_term = torch.zeros_like(f_term)         # drop learned action utility
        logits = alpha * log_p_llm + f_term           # [B,A]
        return logits

    def log_prob(self, feats, log_p_llm, gate_feats) -> torch.Tensor:
        return F.log_softmax(self.forward(feats, log_p_llm, gate_feats), dim=-1)


# ----------------------------------------------------------------------------
# Component 3: ClickModel  examination * attractiveness (+ conditional ad)
# ----------------------------------------------------------------------------
class ClickModel(nn.Module):
    """P_theta(j) propto E_theta(j) * A_theta(j).

    examination  E_theta(j) = sigmoid(g_exam([rank, log_rank, step, focus, expertise...]))
    attractiveness A_theta(j) = sigmoid(g_attr([log p_LLM(j), rel, is_ad, ad_sens, ...]))
    Replaces the fixed  p_LLM^0.1 * pos_bias^0.9 * ad_weight  product.
    """

    def __init__(self, exam_dim: int, attr_dim: int, hidden: int = 32):
        super().__init__()
        self.exam = nn.Sequential(
            nn.Linear(exam_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.attr = nn.Sequential(
            nn.Linear(attr_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, exam_feats: torch.Tensor, attr_feats: torch.Tensor) -> torch.Tensor:
        # feats: [N_results, dim] -> click logits [N_results]
        e = torch.sigmoid(self.exam(exam_feats)).squeeze(-1)
        a = torch.sigmoid(self.attr(attr_feats)).squeeze(-1)
        score = e * a
        return torch.log(torch.clamp(score, min=1e-9))

    def log_prob(self, exam_feats, attr_feats) -> torch.Tensor:
        return F.log_softmax(self.forward(exam_feats, attr_feats), dim=-1)


# ----------------------------------------------------------------------------
# Calibrator: bundle of the trainable components
# ----------------------------------------------------------------------------
class Calibrator(nn.Module):
    """Bundle of all trainable components + featurizer config, saved as one ckpt."""

    def __init__(self, n_intent: int = 3, hidden: int = 64,
                 exam_dim: int = 7, attr_dim: int = 7):
        super().__init__()
        self.n_intent = n_intent
        self.feat_dim = StateFeatures.dim(n_intent)
        self.posterior = ActionPosterior(self.feat_dim, hidden)
        self.click = ClickModel(exam_dim, attr_dim, hidden=32)
        self.featurizer = StateFeaturizer(n_intent)

    # convenience used at inference time (numpy in -> numpy probs out)
    @torch.no_grad()
    def action_probs(self, feats_vec: np.ndarray, log_p_llm: np.ndarray,
                     gate_feats: np.ndarray) -> np.ndarray:
        f = torch.tensor(feats_vec, dtype=torch.float32).unsqueeze(0)
        lp = torch.tensor(log_p_llm, dtype=torch.float32).unsqueeze(0)
        g = torch.tensor(gate_feats, dtype=torch.float32).unsqueeze(0)
        return self.posterior.log_prob(f, lp, g).exp().squeeze(0).numpy()

    @torch.no_grad()
    def click_probs(self, exam_feats: np.ndarray, attr_feats: np.ndarray) -> np.ndarray:
        if ABLATE == "no_click":
            # Drop learned examination*attractiveness; fall back to a 1/rank
            # position-bias prior (rank assumed = row order of the candidates).
            n = exam_feats.shape[0]
            pb = 1.0 / np.arange(1, n + 1, dtype=np.float64)
            return (pb / pb.sum()).astype(np.float32)
        e = torch.tensor(exam_feats, dtype=torch.float32)
        a = torch.tensor(attr_feats, dtype=torch.float32)
        return self.click.log_prob(e, a).exp().numpy()
