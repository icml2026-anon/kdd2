"""Downstream task validation for SearchAgent.

Validates simulation quality by training a click prediction model on simulated
data and evaluating on real data (and vice versa).

Rationale:
    If simulated sessions are realistic, a model trained on simulated data
    should perform comparably to one trained on real data when evaluated on
    held-out real data. This is a stronger validation than aggregate statistics.

Tasks:
    1. Click prediction: Given (query, document_rank, position_features) →
       predict P(click). Train on simulated, test on real.
    2. Abandon prediction: Given session prefix → predict P(abandon).

References:
    - Chuklin et al., "Click Models for Web Search", Morgan & Claypool 2015.
    - Craswell et al., "An Experimental Comparison of Click Position-Bias
      Models", WSDM 2008.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClickSample:
    """A single impression (query, rank) with click label."""
    rank: int
    is_clicked: bool
    session_length: int  # total queries in session
    query_position: int  # which query in session (1-indexed)
    n_prior_clicks: int  # clicks before this impression


@dataclass
class DownstreamResults:
    """Results of downstream click prediction evaluation."""

    # Training data info
    train_source: str  # "simulated" or "real"
    train_n_samples: int
    test_source: str  # "real" or "simulated"
    test_n_samples: int

    # Click prediction metrics
    log_likelihood: float  # mean log-likelihood on test set
    perplexity: float  # exp(-log_likelihood)
    click_accuracy: float  # binary accuracy
    auc: float  # ROC-AUC of predicted click prob vs actual clicks
    rank1_ctr_predicted: float  # predicted CTR at rank 1
    rank1_ctr_actual: float  # actual CTR at rank 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_source": self.train_source,
            "train_n_samples": self.train_n_samples,
            "test_source": self.test_source,
            "test_n_samples": self.test_n_samples,
            "log_likelihood": round(self.log_likelihood, 4),
            "perplexity": round(self.perplexity, 4),
            "click_accuracy": round(self.click_accuracy, 4),
            "auc": round(self.auc, 4),
            "rank1_ctr_predicted": round(self.rank1_ctr_predicted, 4),
            "rank1_ctr_actual": round(self.rank1_ctr_actual, 4),
        }


# ── Position-Based Click Model (PBM) ─────────────────────────────────────

class PositionBasedClickModel:
    """Simple Position-Based Click Model (Craswell et al. 2008).

    P(click | rank) = examination(rank) × attractiveness

    We estimate examination probability per rank via MLE.
    This is a standard baseline click model used in IR evaluation.
    """

    def __init__(self, max_rank: int = 10):
        self.max_rank = max_rank
        self._click_counts = np.zeros(max_rank)
        self._impression_counts = np.zeros(max_rank)
        self._exam_prob = np.ones(max_rank) / max_rank  # uniform prior

    def fit(self, samples: List[ClickSample]) -> None:
        """Fit examination probabilities from click data."""
        self._click_counts = np.zeros(self.max_rank)
        self._impression_counts = np.zeros(self.max_rank)

        for s in samples:
            r = min(s.rank - 1, self.max_rank - 1)  # 0-indexed
            if r >= 0:
                self._impression_counts[r] += 1
                if s.is_clicked:
                    self._click_counts[r] += 1

        # MLE with Laplace smoothing
        self._exam_prob = (self._click_counts + 1) / (self._impression_counts + 2)

    def predict(self, rank: int) -> float:
        """Predict P(click | rank)."""
        r = min(rank - 1, self.max_rank - 1)
        r = max(r, 0)
        return float(self._exam_prob[r])

    def log_likelihood(self, samples: List[ClickSample]) -> float:
        """Compute mean log-likelihood on test samples."""
        if not samples:
            return 0.0

        total_ll = 0.0
        for s in samples:
            p = self.predict(s.rank)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            if s.is_clicked:
                total_ll += np.log(p)
            else:
                total_ll += np.log(1 - p)

        return total_ll / len(samples)

    def get_rank_ctrs(self) -> np.ndarray:
        """Get estimated CTR per rank."""
        return self._exam_prob.copy()


# ── Cascade Click Model (Craswell et al. 2008) ───────────────────────────

class CascadeClickModel:
    """Cascade Click Model (Craswell et al. 2008).

    The user examines results top-down and clicks the first attractive one,
    then stops. P(click at rank r) = attr(r) * prod_{k<r}(1 - attr(k)).
    Per-rank attractiveness is estimated by MLE from observed clicks.

    Like PBM this is a click-only model: it has no notion of query
    reformulation, session abandonment as a decision, satisfaction, or dwell.
    """

    def __init__(self, max_rank: int = 10):
        self.max_rank = max_rank
        self._attr = np.ones(max_rank) / max_rank

    def fit(self, samples: List[ClickSample]) -> None:
        click_counts = np.zeros(self.max_rank)
        imp_counts = np.zeros(self.max_rank)
        for s in samples:
            r = min(s.rank - 1, self.max_rank - 1)
            if r >= 0:
                imp_counts[r] += 1
                if s.is_clicked:
                    click_counts[r] += 1
        # MLE attractiveness per rank with Laplace smoothing
        self._attr = (click_counts + 1) / (imp_counts + 2)

    def predict(self, rank: int) -> float:
        """Marginal P(click at rank) under cascade examination."""
        r = min(max(rank - 1, 0), self.max_rank - 1)
        exam = float(np.prod(1.0 - self._attr[:r]))  # examined iff no click above
        return float(self._attr[r] * exam)

    def log_likelihood(self, samples: List[ClickSample]) -> float:
        if not samples:
            return 0.0
        total = 0.0
        for s in samples:
            p = np.clip(self.predict(s.rank), 1e-7, 1 - 1e-7)
            total += np.log(p) if s.is_clicked else np.log(1 - p)
        return total / len(samples)


# ── Data extraction ───────────────────────────────────────────────────────

def extract_click_samples_from_trec(xml_path: str) -> List[ClickSample]:
    """Extract click samples from TREC Session Track XML."""
    import re
    import xml.etree.ElementTree as ET
    from pathlib import Path

    # TREC 2012 XML contains unescaped '&' (e.g. "merck & co"). Sanitize the
    # same way TRECSERPLoader does (serp.py) so all three years parse.
    raw = Path(xml_path).read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", raw)
    root = ET.fromstring(raw)
    samples = []

    for session_elem in root.findall(".//session"):
        interactions = session_elem.findall("interaction")
        session_length = len(interactions)
        n_prior_clicks = 0

        for q_idx, interaction in enumerate(interactions, 1):
            clicked_ranks = set()
            for click in interaction.findall(".//click"):
                rank_elem = click.find("rank")
                if rank_elem is not None and rank_elem.text:
                    clicked_ranks.add(int(rank_elem.text))

            # Create samples for ranks 1-10 (typical SERP)
            for rank in range(1, 11):
                is_clicked = rank in clicked_ranks
                samples.append(ClickSample(
                    rank=rank,
                    is_clicked=is_clicked,
                    session_length=session_length,
                    query_position=q_idx,
                    n_prior_clicks=n_prior_clicks,
                ))

            n_prior_clicks += len(clicked_ranks)

    return samples


def extract_click_samples_from_sessions(sessions_json_path: str) -> List[ClickSample]:
    """Extract click samples from SearchAgent simulation output."""
    with open(sessions_json_path) as f:
        data = json.load(f)

    samples = []
    for session in data:
        queries = session.get("queries", [session.get("initial_query", "")])
        session_length = len(queries) if isinstance(queries, list) else 1

        steps = session.get("steps", [])
        clicked_ranks_by_query: Dict[int, set] = {}  # query_idx → {ranks}
        current_query_idx = 0

        for step in steps:
            action = step.get("action", "")
            if action in ("reformulate", "new_query"):
                current_query_idx += 1
            # Click actions: current architecture emits "click_result"; older
            # runs used "click_organic"/"click_ad". Accept all so the extractor
            # works across versions.
            if action in ("click_organic", "click_ad", "click_result") and "rank" in step:
                if current_query_idx not in clicked_ranks_by_query:
                    clicked_ranks_by_query[current_query_idx] = set()
                clicked_ranks_by_query[current_query_idx].add(step["rank"])

        n_prior_clicks = 0
        for q_idx in range(session_length):
            clicked = clicked_ranks_by_query.get(q_idx, set())
            for rank in range(1, 11):
                samples.append(ClickSample(
                    rank=rank,
                    is_clicked=rank in clicked,
                    session_length=session_length,
                    query_position=q_idx + 1,
                    n_prior_clicks=n_prior_clicks,
                ))
            n_prior_clicks += len(clicked)

    return samples


# ── Evaluation pipeline ──────────────────────────────────────────────────

def run_downstream_evaluation(
    train_samples: List[ClickSample],
    test_samples: List[ClickSample],
    train_source: str = "simulated",
    test_source: str = "real",
) -> DownstreamResults:
    """Train a click model on train data and evaluate on test data."""
    model = PositionBasedClickModel(max_rank=10)
    model.fit(train_samples)

    ll = model.log_likelihood(test_samples)
    perplexity = np.exp(-ll)

    # Binary accuracy
    correct = 0
    for s in test_samples:
        p = model.predict(s.rank)
        predicted = p >= 0.5
        if predicted == s.is_clicked:
            correct += 1
    accuracy = correct / len(test_samples) if test_samples else 0.0

    # ROC-AUC: how well predicted click prob ranks clicked vs non-clicked
    # impressions. With a pure PBM the only signal is rank, so AUC measures
    # whether the learned position-bias curve discriminates real clicks.
    y_true = [1 if s.is_clicked else 0 for s in test_samples]
    y_score = [model.predict(s.rank) for s in test_samples]
    if test_samples and 0 < sum(y_true) < len(y_true):
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_score))
    else:
        auc = 0.5  # undefined (single-class test set) → chance level

    # Rank 1 CTR comparison
    rank1_pred = model.predict(1)
    rank1_actual_clicks = sum(1 for s in test_samples if s.rank == 1 and s.is_clicked)
    rank1_total = sum(1 for s in test_samples if s.rank == 1)
    rank1_actual = rank1_actual_clicks / rank1_total if rank1_total > 0 else 0.0

    return DownstreamResults(
        train_source=train_source,
        train_n_samples=len(train_samples),
        test_source=test_source,
        test_n_samples=len(test_samples),
        log_likelihood=ll,
        perplexity=perplexity,
        click_accuracy=accuracy,
        auc=auc,
        rank1_ctr_predicted=rank1_pred,
        rank1_ctr_actual=rank1_actual,
    )


def cross_validation_experiment(
    real_samples: List[ClickSample],
    simulated_samples: List[ClickSample],
    seed: int = 42,
    test_fraction: float = 0.3,
) -> Dict[str, DownstreamResults]:
    """Run full cross-validation downstream experiment.

    Trains and evaluates in 4 configurations:
        1. Train real → Test real (oracle upper bound)
        2. Train simulated → Test real (our main validation)
        3. Train simulated → Test simulated (sanity check)
        4. Train real → Test simulated (transfer check)

    Returns:
        Dict mapping config name → DownstreamResults.
    """
    rng = np.random.RandomState(seed)

    # Split real data
    n_test_real = int(len(real_samples) * test_fraction)
    real_idx = rng.permutation(len(real_samples))
    real_test = [real_samples[i] for i in real_idx[:n_test_real]]
    real_train = [real_samples[i] for i in real_idx[n_test_real:]]

    # Split simulated data
    n_test_sim = int(len(simulated_samples) * test_fraction)
    sim_idx = rng.permutation(len(simulated_samples))
    sim_test = [simulated_samples[i] for i in sim_idx[:n_test_sim]]
    sim_train = [simulated_samples[i] for i in sim_idx[n_test_sim:]]

    results = {}

    # 1. Real → Real (oracle)
    results["real_to_real"] = run_downstream_evaluation(
        real_train, real_test, "real", "real"
    )

    # 2. Simulated → Real (main validation)
    results["sim_to_real"] = run_downstream_evaluation(
        sim_train, real_test, "simulated", "real"
    )

    # 3. Simulated → Simulated (sanity)
    results["sim_to_sim"] = run_downstream_evaluation(
        sim_train, sim_test, "simulated", "simulated"
    )

    # 4. Real → Simulated (transfer)
    results["real_to_sim"] = run_downstream_evaluation(
        real_train, sim_test, "real", "simulated"
    )

    return results


def format_downstream_latex(
    results: Dict[str, DownstreamResults],
    caption: str = "Downstream click prediction: cross-training evaluation",
    label: str = "tab:downstream",
) -> str:
    """Format downstream results as a LaTeX table."""
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Train & Test & Log-lik. $\\uparrow$ & Perplexity $\\downarrow$ & "
        "Accuracy & Rank-1 CTR $\\Delta$ \\\\",
        "\\midrule",
    ]

    config_names = {
        "real_to_real": ("Real", "Real"),
        "sim_to_real": ("Simulated", "Real"),
        "sim_to_sim": ("Simulated", "Simulated"),
        "real_to_sim": ("Real", "Simulated"),
    }

    for key in ["real_to_real", "sim_to_real", "sim_to_sim", "real_to_sim"]:
        if key not in results:
            continue
        r = results[key]
        train_name, test_name = config_names[key]
        ctr_delta = abs(r.rank1_ctr_predicted - r.rank1_ctr_actual)

        row = (
            f"{train_name} & {test_name} & {r.log_likelihood:.4f} & "
            f"{r.perplexity:.2f} & {r.click_accuracy:.1%} & {ctr_delta:.4f} \\\\"
        )
        lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{6}{l}{\\footnotesize Sim$\\to$Real close to "
        "Real$\\to$Real indicates high simulation fidelity.} \\\\",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)
