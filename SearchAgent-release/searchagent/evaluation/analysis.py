"""Result analysis module for generating tables, comparisons, and per-trait breakdowns.

Produces:
- LaTeX comparison tables (baselines vs full model vs ablations)
- Per-trait behavioral pattern analysis
- Statistical significance indicators
- Markdown summary reports
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from searchagent.environment.search_env import SessionRecord

logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────────

def load_experiment_summary(path: str) -> Dict[str, Dict[str, Any]]:
    """Load experiment_summary.json from an experiment output directory."""
    with open(path) as f:
        return json.load(f)


def load_sessions(path: str) -> List[SessionRecord]:
    """Load session records from a sessions.json file."""
    from searchagent.environment.search_env import ClickRecord
    with open(path) as f:
        data = json.load(f)

    sessions = []
    for d in data:
        clicks = []
        for step in d.get("steps", []):
            if step.get("action") in ("click_organic", "click_ad") and "rank" in step:
                clicks.append(ClickRecord(
                    step=step.get("step", 0),
                    query=d.get("initial_query", ""),
                    rank=step["rank"],
                    title=step.get("title", ""),
                    is_ad=step.get("is_ad", False),
                    relevance=step.get("relevance", 0.5),
                    satisfaction=step.get("satisfaction", 0.0),
                    dwell_time=step.get("dwell_time", 0.0),
                ))
        sessions.append(SessionRecord(
            persona_id=d["persona_id"],
            initial_query=d["initial_query"],
            steps=d.get("steps", []),
            clicks=clicks,
            queries=d.get("queries", []),
            total_steps=d.get("total_steps", 0),
            abandoned=d.get("abandoned", False),
        ))
    return sessions


# ── LaTeX table generation ───────────────────────────────────────────────

def generate_main_comparison_table(
    summary: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    caption: str = "Behavioral fidelity comparison across agents",
    label: str = "tab:main_comparison",
) -> str:
    """Generate a LaTeX table comparing all agents on key metrics.

    Highlights the best (lowest) value per metric in bold.
    """
    if metrics is None:
        metrics = [
            ("CFS", "CFS", True),
            ("click_position_jsd", "Click JSD", True),
            ("session_length_jsd", "Sess. JSD", True),
            ("clicks_per_session", "Clicks/S $\\Delta$", True),
            ("abandon_rate", "Abandon $\\Delta$", True),
            ("reformulation_rate", "Reform. $\\Delta$", True),
            ("mean_satisfaction", "Satisfaction", False),
        ]

    agents = list(summary.keys())
    n_metrics = len(metrics)

    # Find best values per metric
    best_vals = {}
    for key, display, lower_is_better in metrics:
        vals = []
        for agent in agents:
            m = summary[agent]
            v = m.get(f"{key}_mean", m.get(key, float("inf")))
            if np.isfinite(v):
                vals.append((v, agent))
        if vals:
            if lower_is_better:
                best_vals[key] = min(vals, key=lambda x: x[0])[1]
            else:
                best_vals[key] = max(vals, key=lambda x: x[0])[1]

    # Build LaTeX
    col_spec = "l" + "c" * n_metrics
    lines = [
        f"\\begin{{table}}[ht]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\resizebox{{\\textwidth}}{{!}}{{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\toprule",
    ]

    header = "Agent & " + " & ".join(display for _, display, _ in metrics) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for agent in agents:
        m = summary[agent]
        cells = [_format_agent_name(agent)]
        for key, _, lower_is_better in metrics:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            mean_v = m.get(mean_key, m.get(key, float("nan")))
            std_v = m.get(std_key, 0.0)

            if np.isnan(mean_v) or not np.isfinite(mean_v):
                cell = "--"
            elif std_v > 0:
                cell = f"{mean_v:.4f}$\\pm${std_v:.4f}"
            else:
                cell = f"{mean_v:.4f}"

            if best_vals.get(key) == agent and cell != "--":
                cell = f"\\textbf{{{cell}}}"

            cells.append(cell)

        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_ablation_table(
    summary: Dict[str, Dict[str, Any]],
    caption: str = "Ablation study: contribution of each component",
    label: str = "tab:ablation",
) -> str:
    """Generate a LaTeX table for ablation study results."""
    ablation_agents = [a for a in summary if a.startswith("A") or a == "full"]
    if not ablation_agents:
        return "% No ablation agents found in summary"

    metrics = [
        ("CFS", "CFS", True),
        ("click_position_jsd", "Click JSD", True),
        ("session_length_jsd", "Sess. JSD", True),
        ("clicks_per_session", "Clicks/S $\\Delta$", True),
        ("abandon_rate", "Abandon $\\Delta$", True),
        ("reformulation_rate", "Reform. $\\Delta$", True),
    ]

    # Get full model CFS for delta computation
    full_cfs = summary.get("full", {}).get("CFS_mean", 0)

    col_spec = "l" + "c" * (len(metrics) + 1)
    lines = [
        f"\\begin{{table}}[ht]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\toprule",
    ]

    header = "Variant & " + " & ".join(d for _, d, _ in metrics) + " & $\\Delta$CFS \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for agent in ablation_agents:
        m = summary[agent]
        cells = [_format_agent_name(agent)]

        for key, _, _ in metrics:
            mean_v = m.get(f"{key}_mean", m.get(key, float("nan")))
            if np.isfinite(mean_v):
                cells.append(f"{mean_v:.4f}")
            else:
                cells.append("--")

        # Delta CFS
        agent_cfs = m.get("CFS_mean", m.get("CFS", float("nan")))
        if np.isfinite(agent_cfs) and np.isfinite(full_cfs):
            delta = agent_cfs - full_cfs
            sign = "+" if delta > 0 else ""
            cells.append(f"{sign}{delta:.4f}")
        else:
            cells.append("--")

        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_descriptive_table(
    summary: Dict[str, Dict[str, Any]],
    caption: str = "Descriptive statistics of simulated sessions",
    label: str = "tab:descriptive",
) -> str:
    """Generate a LaTeX table with descriptive session statistics."""
    agents = list(summary.keys())
    stats = [
        ("mean_session_length", "Avg. Length"),
        ("mean_clicks", "Avg. Clicks"),
        ("mean_queries", "Avg. Queries"),
        ("abandon_rate_raw", "Abandon Rate"),
        ("mean_satisfaction", "Satisfaction"),
    ]

    col_spec = "l" + "c" * len(stats)
    lines = [
        f"\\begin{{table}}[ht]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\toprule",
    ]

    header = "Agent & " + " & ".join(d for _, d in stats) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for agent in agents:
        m = summary[agent]
        cells = [_format_agent_name(agent)]
        for key, _ in stats:
            v = m.get(f"{key}_mean", m.get(key, float("nan")))
            if np.isfinite(v):
                if v < 1 and "rate" in key.lower():
                    cells.append(f"{v:.1%}")
                else:
                    cells.append(f"{v:.2f}")
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_trait_table(
    trait_analysis: Dict[str, Dict[str, Dict[str, float]]],
    trait_name: str,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """Generate a LaTeX table for per-trait behavioral analysis."""
    if trait_name not in trait_analysis:
        return f"% No data for trait: {trait_name}"

    if caption is None:
        caption = f"Behavioral patterns by {trait_name.replace('_', ' ')}"
    if label is None:
        label = f"tab:trait_{trait_name}"

    data = trait_analysis[trait_name]
    trait_vals = sorted(data.keys())

    stat_keys = [
        ("n_sessions", "N"),
        ("avg_clicks", "Clicks"),
        ("avg_ad_clicks", "Ad Clicks"),
        ("avg_session_length", "Length"),
        ("abandon_rate", "Abandon"),
        ("reformulation_rate", "Reform."),
    ]

    col_spec = "l" + "c" * len(stat_keys)
    lines = [
        f"\\begin{{table}}[ht]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\toprule",
    ]

    header = f"{trait_name.replace('_', ' ').title()} & " + " & ".join(d for _, d in stat_keys) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for tv in trait_vals:
        cells = [tv.replace("_", "\\_")]
        for key, _ in stat_keys:
            v = data[tv].get(key, 0)
            if isinstance(v, int):
                cells.append(str(v))
            elif "rate" in key:
                cells.append(f"{v:.1%}")
            else:
                cells.append(f"{v:.2f}")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# ── Markdown report ──────────────────────────────────────────────────────

def generate_markdown_report(
    summary: Dict[str, Dict[str, Any]],
) -> str:
    """Generate a readable Markdown analysis report."""
    lines = ["# SearchAgent Experiment Report\n"]

    # Overall comparison
    lines.append("## Agent Comparison\n")
    lines.append("| Agent | CFS ↓ | Click JSD ↓ | Session JSD ↓ | Clicks/S Δ ↓ | Abandon Δ ↓ | Reform. Δ ↓ |")
    lines.append("|-------|-------|------------|-------------|------------|------------|------------|")

    for agent, m in summary.items():
        cfs = m.get("CFS_mean", m.get("CFS", 0))
        cjsd = m.get("click_position_jsd_mean", m.get("click_position_jsd", 0))
        sjsd = m.get("session_length_jsd_mean", m.get("session_length_jsd", 0))
        cpsg = m.get("clicks_per_session_mean", m.get("clicks_per_session", 0))
        abng = m.get("abandon_rate_mean", m.get("abandon_rate", 0))
        refg = m.get("reformulation_rate_mean", m.get("reformulation_rate", 0))
        lines.append(
            f"| {agent} | {cfs:.4f} | {cjsd:.4f} | {sjsd:.4f} | "
            f"{cpsg:.4f} | {abng:.4f} | {refg:.4f} |"
        )

    lines.append("")

    # Per-trait analysis (for full model)
    full_data = summary.get("full", {})
    trait_analysis = full_data.get("trait_analysis", {})
    if trait_analysis:
        lines.append("## Per-Trait Behavioral Patterns (Full Model)\n")
        for trait_name, trait_data in trait_analysis.items():
            lines.append(f"### {trait_name.replace('_', ' ').title()}\n")
            lines.append("| Value | N | Clicks | Ad Clicks | Length | Abandon | Reform. |")
            lines.append("|-------|---|--------|-----------|--------|---------|---------|")
            for tv, stats in sorted(trait_data.items()):
                lines.append(
                    f"| {tv} | {stats['n_sessions']} | {stats['avg_clicks']:.1f} | "
                    f"{stats['avg_ad_clicks']:.2f} | {stats['avg_session_length']:.1f} | "
                    f"{stats['abandon_rate']:.1%} | {stats['reformulation_rate']:.1%} |"
                )
            lines.append("")

    # Descriptive stats
    lines.append("## Descriptive Statistics\n")
    lines.append("| Agent | Avg. Length | Avg. Clicks | Avg. Queries | Abandon Rate | Satisfaction |")
    lines.append("|-------|-----------|-----------|-------------|-------------|-------------|")
    for agent, m in summary.items():
        length = m.get("mean_session_length_mean", m.get("mean_session_length", 0))
        clicks = m.get("mean_clicks_mean", m.get("mean_clicks", 0))
        queries = m.get("mean_queries_mean", m.get("mean_queries", 0))
        abn = m.get("abandon_rate_raw_mean", m.get("abandon_rate_raw", 0))
        sat = m.get("mean_satisfaction_mean", m.get("mean_satisfaction", 0))
        lines.append(
            f"| {agent} | {length:.1f} | {clicks:.1f} | {queries:.1f} | "
            f"{abn:.1%} | {sat:.2f} |"
        )
    lines.append("")

    # Key findings
    lines.append("## Key Findings\n")
    agents_by_cfs = sorted(
        summary.items(),
        key=lambda x: x[1].get("CFS_mean", x[1].get("CFS", float("inf")))
    )
    if agents_by_cfs:
        best_agent = agents_by_cfs[0][0]
        best_cfs = agents_by_cfs[0][1].get("CFS_mean", agents_by_cfs[0][1].get("CFS", 0))
        lines.append(f"- **Best CFS**: {best_agent} ({best_cfs:.4f})")

    if trait_analysis.get("patience_level"):
        pat_data = trait_analysis["patience_level"]
        for level in ["low", "high"]:
            if level in pat_data:
                lines.append(f"- **Patience={level}**: abandon rate = {pat_data[level].get('abandon_rate', 0):.1%}, "
                             f"avg length = {pat_data[level].get('avg_session_length', 0):.1f}")

    return "\n".join(lines)


# ── Helper ───────────────────────────────────────────────────────────────

def _format_agent_name(name: str) -> str:
    """Format agent name for LaTeX display."""
    mapping = {
        "full": "SearchAgent (Full)",
        "random": "Random",
        "rule_based": "Rule-Based",
        "naive_llm": "Naive LLM",
        "A1_no_persona": "w/o Persona",
        "A2_no_intent": "w/o Intent",
        "A3_no_memory": "w/o Memory",
        "A4_no_ad": "w/o Ad Model",
        "A5_no_posbias": "w/o Pos. Bias",
    }
    return mapping.get(name, name)


def save_all_tables(
    summary_path: str,
    output_dir: str,
) -> None:
    """Load experiment summary and save all LaTeX tables and markdown report."""
    summary = load_experiment_summary(summary_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Main comparison table
    main_table = generate_main_comparison_table(summary)
    (out / "main_comparison.tex").write_text(main_table)

    # Ablation table
    ablation_table = generate_ablation_table(summary)
    (out / "ablation.tex").write_text(ablation_table)

    # Descriptive table
    desc_table = generate_descriptive_table(summary)
    (out / "descriptive.tex").write_text(desc_table)

    # Per-trait tables (from full model)
    full_data = summary.get("full", {})
    trait_analysis = full_data.get("trait_analysis", {})
    for trait_name in trait_analysis:
        table = generate_trait_table(trait_analysis, trait_name)
        (out / f"trait_{trait_name}.tex").write_text(table)

    # Bootstrap CIs for summary table annotation
    # Check if CFS_ci_lower exists (populated by run_experiment.py)
    has_ci = any("CFS_ci_lower" in v for v in summary.values() if isinstance(v, dict))
    if has_ci:
        ci_lines = ["% Bootstrap 95% confidence intervals for CFS"]
        for agent, data in summary.items():
            ci_lo = data.get("CFS_ci_lower", "")
            ci_hi = data.get("CFS_ci_upper", "")
            if ci_lo and ci_hi:
                ci_lines.append(f"% {agent}: [{ci_lo:.4f}, {ci_hi:.4f}]")
        (out / "cfs_confidence_intervals.tex").write_text("\n".join(ci_lines))

    # Markdown report
    report = generate_markdown_report(summary)
    (out / "report.md").write_text(report)

    logger.info("All tables saved to %s", out)
