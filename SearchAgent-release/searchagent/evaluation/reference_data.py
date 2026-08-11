from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


_METRICS = (
    "click_position_jsd",
    "session_length_jsd",
    "clicks_per_session",
    "abandon_rate",
    "reformulation_rate",
)


def _normalize(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    total = float(y.sum())
    if total <= 0:
        return np.full(len(y), 1.0 / max(len(y), 1), dtype=np.float64)
    return y / total


def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalize(p)
    q = _normalize(q)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log((p + eps) / (m + eps))) + 0.5 * np.sum(q * np.log((q + eps) / (m + eps))))


@dataclass
class SearchLogReference:
    click_position_dist: np.ndarray = field(default_factory=lambda: np.array([0.6178, 0.2309, 0.0878, 0.0356, 0.0166, 0.0065, 0.0030, 0.0006, 0.0006, 0.0006], dtype=np.float64))
    session_length_dist: np.ndarray = field(default_factory=lambda: np.array([0.3795, 0.1543, 0.1472, 0.1154, 0.0851, 0.0422, 0.0334, 0.0255, 0.0040, 0.0135], dtype=np.float64))
    clicks_per_session: float = 1.34
    queries_per_session: float = 2.90
    abandon_rate: float = 0.49
    reformulation_rate: float = 0.33
    dwell_time_mean: float = 31.5
    dwell_time_median: float = 22.7
    bootstrap_scales: Dict[str, float] = field(default_factory=lambda: {name: 1.0 for name in _METRICS})
    bootstrap_replicates: int = 0

    def __post_init__(self):
        self.click_position_dist = _normalize(self.click_position_dist)
        self.session_length_dist = _normalize(self.session_length_dist)
        self.bootstrap_scales = {name: max(float(self.bootstrap_scales.get(name, 1.0)), 1e-8) for name in _METRICS}


REFERENCE = SearchLogReference()


def _bootstrap_scales(trec_sessions, n_bootstrap: int = 1000, seed: int = 314159) -> Dict[str, float]:
    n = len(trec_sessions)
    rank_rows = np.zeros((n, 10), dtype=np.float64)
    length_rows = np.zeros((n, 10), dtype=np.float64)
    clicks = np.zeros(n, dtype=np.float64)
    abandoned = np.zeros(n, dtype=np.float64)
    reformulated = np.zeros(n, dtype=np.float64)
    for i, session in enumerate(trec_sessions):
        for rank in session.click_positions:
            if rank > 0:
                rank_rows[i, min(int(rank) - 1, 9)] += 1.0
        length_rows[i, min(max(int(session.num_queries), 1) - 1, 9)] = 1.0
        clicks[i] = float(session.total_clicks)
        abandoned[i] = 0.0 if session.has_clicks else 1.0
        reformulated[i] = 1.0 if session.num_queries > 1 else 0.0
    rng = np.random.default_rng(seed)
    values = {name: [] for name in _METRICS}
    for _ in range(max(int(n_bootstrap), 1)):
        ia = rng.integers(0, n, size=n)
        ib = rng.integers(0, n, size=n)
        values["click_position_jsd"].append(_jsd(rank_rows[ia].sum(axis=0), rank_rows[ib].sum(axis=0)))
        values["session_length_jsd"].append(_jsd(length_rows[ia].sum(axis=0), length_rows[ib].sum(axis=0)))
        values["clicks_per_session"].append(abs(float(clicks[ia].mean() - clicks[ib].mean())))
        values["abandon_rate"].append(abs(float(abandoned[ia].mean() - abandoned[ib].mean())))
        values["reformulation_rate"].append(abs(float(reformulated[ia].mean() - reformulated[ib].mean())))
    return {name: max(float(np.mean(vals)), 1e-8) for name, vals in values.items()}


def compute_reference_from_sessions(trec_sessions, n_bootstrap: int = 1000, seed: int = 314159) -> SearchLogReference:
    if not trec_sessions:
        return SearchLogReference()
    n = len(trec_sessions)
    rank_counts = np.zeros(10, dtype=np.float64)
    length_counts = np.zeros(10, dtype=np.float64)
    total_clicks = 0
    all_dwells = []
    for session in trec_sessions:
        for rank in session.click_positions:
            if rank > 0:
                rank_counts[min(int(rank) - 1, 9)] += 1.0
        length_counts[min(max(int(session.num_queries), 1) - 1, 9)] += 1.0
        total_clicks += int(session.total_clicks)
        for interaction in session.interactions:
            for click in interaction.clicks:
                if click.dwell_time > 0:
                    all_dwells.append(float(click.dwell_time))
    scales = _bootstrap_scales(trec_sessions, n_bootstrap=n_bootstrap, seed=seed)
    return SearchLogReference(
        click_position_dist=rank_counts if rank_counts.sum() > 0 else REFERENCE.click_position_dist.copy(),
        session_length_dist=length_counts,
        clicks_per_session=float(total_clicks / n),
        queries_per_session=float(np.mean([session.num_queries for session in trec_sessions])),
        abandon_rate=float(np.mean([not session.has_clicks for session in trec_sessions])),
        reformulation_rate=float(np.mean([session.num_queries > 1 for session in trec_sessions])),
        dwell_time_mean=float(np.mean(all_dwells)) if all_dwells else 0.0,
        dwell_time_median=float(np.median(all_dwells)) if all_dwells else 0.0,
        bootstrap_scales=scales,
        bootstrap_replicates=max(int(n_bootstrap), 1),
    )
