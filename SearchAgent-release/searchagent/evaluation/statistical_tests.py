"""Statistical significance testing for SearchAgent evaluation.

Provides:
    - Bootstrap confidence intervals for individual metrics
    - Paired Wilcoxon signed-rank test between agent pairs
    - Permutation test (non-parametric, valid for small n)
    - Effect size (Cohen's d)
    - Significance markers for LaTeX tables

Usage:
    from searchagent.evaluation.statistical_tests import (
        bootstrap_ci,
        paired_permutation_test,
        wilcoxon_test,
        cohens_d,
        significance_marker,
        pairwise_significance_table,
    )

References:
    - Efron & Tibshirani, "An Introduction to the Bootstrap", 1993.
    - Good, "Permutation, Parametric and Bootstrap Tests of Hypotheses", 2005.
    - Dror et al., "The Hitchhiker's Guide to Testing Statistical Significance
      in NLP", ACL 2018.  (recommends permutation/bootstrap for small n in NLP/IR)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


# ── Bootstrap confidence intervals ────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        values: 1-D array of observed metric values (one per seed/run).
        confidence: Confidence level (default 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        m = float(values[0]) if n == 1 else float("nan")
        return m, m, m

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(values.mean()), ci_lower, ci_upper


# ── Paired permutation test ───────────────────────────────────────────────

def paired_permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
    alternative: str = "two-sided",
) -> float:
    """Paired permutation test for difference in means.

    Recommended by Dror et al. (ACL 2018) for small sample sizes in NLP/IR.

    Args:
        values_a: Metric values for agent A (one per seed).
        values_b: Metric values for agent B (one per seed).
        n_permutations: Number of random permutations.
        seed: Random seed.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        p-value.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    assert len(a) == len(b), "Arrays must have equal length (paired test)"
    n = len(a)

    if n < 2:
        return 1.0

    diffs = a - b
    observed = np.abs(diffs.mean()) if alternative == "two-sided" else diffs.mean()

    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_mean = (diffs * signs).mean()
        if alternative == "two-sided":
            if np.abs(perm_mean) >= observed:
                count += 1
        elif alternative == "less":
            if perm_mean <= observed:
                count += 1
        else:  # greater
            if perm_mean >= observed:
                count += 1

    return (count + 1) / (n_permutations + 1)


# ── Wilcoxon signed-rank test ─────────────────────────────────────────────

def wilcoxon_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alternative: str = "two-sided",
) -> float:
    """Paired Wilcoxon signed-rank test.

    Falls back to permutation test if n < 6 (Wilcoxon unreliable).

    Returns:
        p-value.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n = len(a)

    if n < 2:
        return 1.0

    # Wilcoxon requires n >= 6 for meaningful results
    if n < 6:
        return paired_permutation_test(a, b, alternative=alternative)

    try:
        _, p = scipy_stats.wilcoxon(a, b, alternative=alternative)
        return float(p)
    except ValueError:
        # All differences are zero
        return 1.0


# ── Effect size ───────────────────────────────────────────────────────────

def cohens_d(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> float:
    """Compute Cohen's d effect size for paired observations.

    Interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large

    Returns:
        Cohen's d (positive means A > B).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diff = a - b
    if diff.std() == 0:
        return 0.0
    return float(diff.mean() / diff.std())


# ── Significance markers ──────────────────────────────────────────────────

def significance_marker(p_value: float) -> str:
    """Return a significance marker for LaTeX tables.

    Convention (common in IR/NLP):
        p < 0.001: ***  (highly significant)
        p < 0.01:  **   (very significant)
        p < 0.05:  *    (significant)
        p ≥ 0.05:  (empty)
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


def significance_marker_latex(p_value: float) -> str:
    """Return significance markers using LaTeX dagger notation.

    Convention:
        p < 0.01: $^{\\ddagger}$
        p < 0.05: $^{\\dagger}$
        p ≥ 0.05: (empty)
    """
    if p_value < 0.01:
        return "$^{\\ddagger}$"
    elif p_value < 0.05:
        return "$^{\\dagger}$"
    return ""


# ── Pairwise significance table ──────────────────────────────────────────

@dataclass
class PairwiseResult:
    """Result of a pairwise significance test."""
    agent_a: str
    agent_b: str
    metric: str
    mean_a: float
    mean_b: float
    diff: float
    p_value: float
    effect_size: float
    significant: bool


def pairwise_significance_table(
    per_seed_results: Dict[str, List[Dict[str, float]]],
    metric: str = "CFS",
    baseline: Optional[str] = None,
    alpha: float = 0.05,
) -> List[PairwiseResult]:
    """Run pairwise significance tests between agents.

    Args:
        per_seed_results: {agent_name: [seed1_metrics, seed2_metrics, ...]}
            Each element is a dict of metric_name → value.
        metric: Which metric to test.
        baseline: If provided, only test each agent against this baseline.
            If None, test all pairs.
        alpha: Significance threshold.

    Returns:
        List of PairwiseResult objects.
    """
    agents = list(per_seed_results.keys())
    results = []

    if baseline is not None:
        pairs = [(baseline, a) for a in agents if a != baseline]
    else:
        pairs = list(combinations(agents, 2))

    for agent_a, agent_b in pairs:
        vals_a = np.array([d.get(metric, float("nan")) for d in per_seed_results[agent_a]])
        vals_b = np.array([d.get(metric, float("nan")) for d in per_seed_results[agent_b]])

        # Skip if any NaN
        mask = np.isfinite(vals_a) & np.isfinite(vals_b)
        if mask.sum() < 2:
            continue

        vals_a = vals_a[mask]
        vals_b = vals_b[mask]

        p = paired_permutation_test(vals_a, vals_b)
        d = cohens_d(vals_a, vals_b)

        results.append(PairwiseResult(
            agent_a=agent_a,
            agent_b=agent_b,
            metric=metric,
            mean_a=float(vals_a.mean()),
            mean_b=float(vals_b.mean()),
            diff=float(vals_a.mean() - vals_b.mean()),
            p_value=p,
            effect_size=d,
            significant=p < alpha,
        ))

    return results


def format_significance_table_latex(
    results: List[PairwiseResult],
    caption: str = "Pairwise significance tests",
    label: str = "tab:significance",
) -> str:
    """Format pairwise test results as a LaTeX table."""
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Agent A & Agent B & Mean A & Mean B & $\\Delta$ & $p$-value & $d$ \\\\",
        "\\midrule",
    ]

    for r in results:
        sig = significance_marker_latex(r.p_value)
        lines.append(
            f"{r.agent_a} & {r.agent_b} & {r.mean_a:.4f} & {r.mean_b:.4f} & "
            f"{r.diff:+.4f}{sig} & {r.p_value:.4f} & {r.effect_size:.2f} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{7}{l}{\\footnotesize $^\\dagger p<0.05$, $^\\ddagger p<0.01$. "
        "Effect size $d$: $<$0.2 negligible, 0.2--0.5 small, 0.5--0.8 medium, $\\geq$0.8 large.} \\\\",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)
