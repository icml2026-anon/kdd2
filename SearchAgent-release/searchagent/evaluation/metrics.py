"""Behavioral fidelity metrics for evaluating simulated search sessions.

Five core metrics grounded in real search log studies, with a normalized
Composite Fidelity Score (CFS).

References:
    [1] Joachims et al., "Accurately Interpreting Clickthrough Data as
        Implicit Feedback", SIGIR 2005.
    [2] Craswell et al., "An Experimental Comparison of Click Position-Bias
        Models", WSDM 2008.
    [3] Jansen & Spink, "How Are We Searching the World Wide Web?", IPM 2006.
    [4] Huang & Efthimiadis, "Analyzing and Evaluating Query Reformulation
        Strategies in Web Search Logs", CIKM 2009.
    [5] Li et al., "Good Abandonment in Mobile and PC Internet Search",
        SIGIR 2009.
    [6] Silverstein et al., "Analysis of a Very Large Web Search Engine Query
        Log", SIGIR Forum, 1999.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import entropy

from searchagent.environment.search_env import SessionRecord
from searchagent.evaluation.reference_data import REFERENCE

logger = logging.getLogger(__name__)


# ── Helper functions ─────────────────────────────────────────────────────

def _normalize_distribution(counts: np.ndarray, smoothing: float = 1e-8) -> np.ndarray:
    """Normalize counts to a probability distribution with Laplace smoothing."""
    dist = counts.astype(np.float64) + smoothing
    return dist / dist.sum()


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, ln2])."""
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m) + 0.5 * entropy(q, m))


# ── Metric 1: Click Position JSD ────────────────────────────────────────

def click_position_jsd(
    simulated: List[SessionRecord],
    reference: Optional[np.ndarray] = None,
) -> float:
    """Jensen-Shannon divergence of click position distributions.

    Reference: Joachims et al. 2005 + Craswell et al. 2008.
    JSD is preferred over KL: symmetric and bounded in [0, ln2 ≈ 0.693].
    """
    ref = reference if reference is not None else REFERENCE.click_position_dist
    max_rank = len(ref)

    sim_counts = np.zeros(max_rank)
    for session in simulated:
        for pos in session.click_positions:
            idx = min(pos - 1, max_rank - 1)
            if 0 <= idx < max_rank:
                sim_counts[idx] += 1

    if sim_counts.sum() == 0:
        return np.log(2)  # max JSD when no clicks

    sim_dist = _normalize_distribution(sim_counts)
    ref_dist = _normalize_distribution(ref)
    return _jsd(sim_dist, ref_dist)


# ── Metric 2: Session Length JSD ─────────────────────────────────────────

def session_length_jsd(
    simulated: List[SessionRecord],
    reference: Optional[np.ndarray] = None,
) -> float:
    """Jensen-Shannon divergence of session length distributions.

    Reference: Jansen et al. 2000, Silverstein et al. 1999.
    """
    ref = reference if reference is not None else REFERENCE.session_length_dist
    max_len = len(ref)

    sim_counts = np.zeros(max_len)
    for session in simulated:
        idx = min(session.session_length - 1, max_len - 1)
        if 0 <= idx < max_len:
            sim_counts[idx] += 1

    if sim_counts.sum() == 0:
        return np.log(2)

    sim_dist = _normalize_distribution(sim_counts)
    ref_dist = _normalize_distribution(ref)
    return _jsd(sim_dist, ref_dist)


# ── Metric 3: Clicks per Session Gap ────────────────────────────────────

def clicks_per_session_gap(
    simulated: List[SessionRecord],
    reference: Optional[float] = None,
) -> float:
    """Absolute difference between simulated and reference mean clicks/session.

    Reference: ~1.6 clicks/session (Jansen & Spink 2006).
    Normalized by reference value to make it scale-independent.
    """
    ref = reference if reference is not None else REFERENCE.clicks_per_session
    if not simulated:
        return 1.0

    sim_mean = sum(len(s.clicks) for s in simulated) / len(simulated)
    return abs(sim_mean - ref) / ref  # normalized: 0 = perfect, 1 = 100% off


# ── Metric 4: Abandon Rate Gap ──────────────────────────────────────────

def abandon_rate_gap(
    simulated: List[SessionRecord],
    reference: Optional[float] = None,
) -> float:
    """Absolute difference between simulated and reference abandon rate.

    Reference: ~35% session abandonment (Li et al. 2009, includes good abandonment).
    """
    ref = reference if reference is not None else REFERENCE.abandon_rate
    if not simulated:
        return ref

    abandoned = sum(1 for s in simulated if s.abandoned)
    sim_rate = abandoned / len(simulated)
    return abs(sim_rate - ref)


# ── Metric 5: Reformulation Rate Gap ────────────────────────────────────

def reformulation_rate_gap(
    simulated: List[SessionRecord],
    reference: Optional[float] = None,
) -> float:
    """Absolute difference between simulated and reference reformulation rate.

    Reference: ~33% of sessions contain reformulation (Huang & Efthimiadis 2009).
    """
    ref = reference if reference is not None else REFERENCE.reformulation_rate
    if not simulated:
        return ref

    sessions_with_reform = sum(1 for s in simulated if s.num_queries > 1)
    sim_rate = sessions_with_reform / len(simulated)
    return abs(sim_rate - ref)


# ── Composite Fidelity Score ─────────────────────────────────────────────

def composite_fidelity_score(
    simulated: List[SessionRecord],
    weights: Optional[Dict[str, float]] = None,
    reference=None,
) -> Dict[str, float]:
    """Compute the Composite Fidelity Score (CFS) and all individual metrics.

    Five core metrics, each normalized to [0, 1]:
      1. Click Position JSD  (distributional, bounded [0, ln2])
      2. Session Length JSD   (distributional, bounded [0, ln2])
      3. Clicks/Session Gap   (rate, normalized by reference)
      4. Abandon Rate Gap     (rate, absolute difference)
      5. Reformulation Rate Gap (rate, absolute difference)

    Lower CFS = better fidelity to real-world search behavior.
    CFS ∈ [0, 1] by construction.

    Args:
        reference: A SearchLogReference instance for per-dataset comparison.
                   If None, uses the default TREC 2014 reference.
    """
    if reference is None:
        reference = REFERENCE

    if weights is None:
        weights = {
            "click_position_jsd": 0.30,
            "clicks_per_session": 0.70,
        }

    # Compute raw metrics using the provided reference
    raw = {
        "click_position_jsd": click_position_jsd(simulated, reference=reference.click_position_dist),
        "session_length_jsd": session_length_jsd(simulated, reference=reference.session_length_dist),
        "clicks_per_session": clicks_per_session_gap(simulated, reference=reference.clicks_per_session),
        "abandon_rate": abandon_rate_gap(simulated, reference=reference.abandon_rate),
        "reformulation_rate": reformulation_rate_gap(simulated, reference=reference.reformulation_rate),
    }

    # Normalization: map each metric to [0, 1]
    # JSD is already bounded in [0, ln2]; gaps are capped at 1.0
    _caps = {
        "click_position_jsd": np.log(2),   # ~0.693
        "session_length_jsd": np.log(2),
        "clicks_per_session": 2.0,          # >200% deviation → max penalty
        "abandon_rate": 0.5,                # >50pp deviation → max penalty
        "reformulation_rate": 0.5,
    }

    results = {}
    cfs = 0.0
    for metric, val in raw.items():
        cap = _caps[metric]
        normalized = min(val, cap) / cap if np.isfinite(val) else 1.0
        results[metric] = val
        results[f"{metric}_norm"] = normalized
        if metric in weights:
            cfs += weights[metric] * normalized

    results["CFS"] = cfs

    # Descriptive statistics (for analysis, not part of CFS)
    if simulated:
        lengths = [s.session_length for s in simulated]
        n_clicks = [len(s.clicks) for s in simulated]
        n_queries = [s.num_queries for s in simulated]
        results["mean_session_length"] = float(np.mean(lengths))
        results["mean_clicks"] = float(np.mean(n_clicks))
        results["mean_queries"] = float(np.mean(n_queries))
        results["abandon_rate_raw"] = sum(1 for s in simulated if s.abandoned) / len(simulated)
        results["mean_satisfaction"] = float(np.mean(
            [s.mean_satisfaction for s in simulated if s.mean_satisfaction > 0]
        )) if any(s.mean_satisfaction > 0 for s in simulated) else 0.0
    else:
        results["mean_session_length"] = 0.0
        results["mean_clicks"] = 0.0
        results["mean_queries"] = 0.0
        results["abandon_rate_raw"] = 0.0
        results["mean_satisfaction"] = 0.0

    results["num_sessions"] = len(simulated)
    return results
