from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from searchagent.environment.search_env import SessionRecord
from searchagent.evaluation.reference_data import REFERENCE, SearchLogReference


def _normalize_distribution(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    if float(x.sum()) <= 0:
        return np.full(len(x), 1.0 / max(len(x), 1), dtype=np.float64)
    x = np.maximum(x, 0.0) + eps
    return x / x.sum()


def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalize_distribution(p, eps)
    q = _normalize_distribution(q, eps)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log((p + eps) / (m + eps))) + 0.5 * np.sum(q * np.log((q + eps) / (m + eps))))


def click_position_jsd(simulated: List[SessionRecord], reference: Optional[np.ndarray] = None) -> float:
    ref = np.asarray(reference if reference is not None else REFERENCE.click_position_dist, dtype=np.float64)
    counts = np.zeros(len(ref), dtype=np.float64)
    for session in simulated:
        for rank in session.click_positions:
            if rank > 0:
                counts[min(int(rank) - 1, len(ref) - 1)] += 1.0
    if counts.sum() <= 0:
        return float(np.log(2.0))
    return _jsd(counts, ref)


def session_length_jsd(simulated: List[SessionRecord], reference: Optional[np.ndarray] = None) -> float:
    ref = np.asarray(reference if reference is not None else REFERENCE.session_length_dist, dtype=np.float64)
    counts = np.zeros(len(ref), dtype=np.float64)
    for session in simulated:
        length = max(int(session.session_length), 1)
        counts[min(length - 1, len(ref) - 1)] += 1.0
    if counts.sum() <= 0:
        return float(np.log(2.0))
    return _jsd(counts, ref)


def clicks_per_session_gap(simulated: List[SessionRecord], reference: Optional[float] = None) -> float:
    ref = float(reference if reference is not None else REFERENCE.clicks_per_session)
    if not simulated:
        return abs(ref)
    return abs(float(np.mean([len(session.clicks) for session in simulated])) - ref)


def abandon_rate_gap(simulated: List[SessionRecord], reference: Optional[float] = None) -> float:
    ref = float(reference if reference is not None else REFERENCE.abandon_rate)
    if not simulated:
        return abs(ref)
    return abs(float(np.mean([session.abandoned for session in simulated])) - ref)


def reformulation_rate_gap(simulated: List[SessionRecord], reference: Optional[float] = None) -> float:
    ref = float(reference if reference is not None else REFERENCE.reformulation_rate)
    if not simulated:
        return abs(ref)
    return abs(float(np.mean([session.num_queries > 1 for session in simulated])) - ref)



def replay_alignment_metrics(simulated: List[SessionRecord], real_sessions: List) -> Dict[str, float]:
    by_id = {int(session.persona_id): session for session in simulated}
    matched = 0
    generated_clicks = 0
    real_clicks = 0
    abandon_matches = 0
    for real in real_sessions:
        generated = by_id.get(int(real.session_num))
        generated_ranks = [] if generated is None else list(generated.click_positions)
        real_ranks = list(real.click_positions)
        generated_counter = Counter(generated_ranks)
        real_counter = Counter(real_ranks)
        matched += sum(min(count, real_counter.get(rank, 0)) for rank, count in generated_counter.items())
        generated_clicks += len(generated_ranks)
        real_clicks += len(real_ranks)
        abandon_matches += int((len(generated_ranks) == 0) == (len(real_ranks) == 0))
    precision = matched / generated_clicks if generated_clicks else 0.0
    recall = matched / real_clicks if real_clicks else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    agreement = abandon_matches / len(real_sessions) if real_sessions else 0.0
    return {
        "replay_precision": float(precision),
        "replay_recall": float(recall),
        "replay_f1": float(f1),
        "abandonment_agreement": float(agreement),
    }

def population_fidelity_discrepancy(simulated: List[SessionRecord], reference: Optional[SearchLogReference] = None, eps: float = 1e-8) -> Dict[str, float]:
    ref = reference if reference is not None else REFERENCE
    raw = {
        "click_position_jsd": click_position_jsd(simulated, ref.click_position_dist),
        "session_length_jsd": session_length_jsd(simulated, ref.session_length_dist),
        "clicks_per_session": clicks_per_session_gap(simulated, ref.clicks_per_session),
        "abandon_rate": abandon_rate_gap(simulated, ref.abandon_rate),
        "reformulation_rate": reformulation_rate_gap(simulated, ref.reformulation_rate),
    }
    normalized = {name: float(value / (ref.bootstrap_scales.get(name, 1.0) + eps)) for name, value in raw.items()}
    result = dict(raw)
    result.update({f"{name}_norm": value for name, value in normalized.items()})
    result["PFD"] = float(np.mean(list(normalized.values())))
    result["CFS"] = result["PFD"]
    if simulated:
        result["mean_session_length"] = float(np.mean([session.session_length for session in simulated]))
        result["mean_clicks"] = float(np.mean([len(session.clicks) for session in simulated]))
        result["mean_queries"] = float(np.mean([session.num_queries for session in simulated]))
        result["abandon_rate_raw"] = float(np.mean([session.abandoned for session in simulated]))
        sats = [session.mean_satisfaction for session in simulated if session.mean_satisfaction > 0]
        result["mean_satisfaction"] = float(np.mean(sats)) if sats else 0.0
    else:
        result["mean_session_length"] = 0.0
        result["mean_clicks"] = 0.0
        result["mean_queries"] = 0.0
        result["abandon_rate_raw"] = 0.0
        result["mean_satisfaction"] = 0.0
    result["num_sessions"] = len(simulated)
    return result


def composite_fidelity_score(simulated: List[SessionRecord], weights: Optional[Dict[str, float]] = None, reference=None) -> Dict[str, float]:
    return population_fidelity_discrepancy(simulated, reference=reference)
