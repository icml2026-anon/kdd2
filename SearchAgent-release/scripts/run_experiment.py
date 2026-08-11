"""Unified experiment runner for SearchAgent and the paper baseline set.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml \
        --agents full_pp react polyskill reflexion cot tot usersimcrs_v2 cshi usimagent bases agent4rec simuser convsearch_r1 \
        --num_sessions 50 --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from searchagent.agent.baselines import (
    Agent4RecAgent,
    BASESAgent,
    CSHIAgent,
    CoTAgent,
    ConvSearchR1Agent,
    NaiveLLMAgent,
    PolySkillAgent,
    RandomAgent,
    ReActSearchAgent,
    ReflexionAgent,
    RuleBasedAgent,
    SimUSERAgent,
    TreeOfThoughtsAgent,
    USimAgent,
    UserSimCRSv2Agent,
)
from searchagent.agent.llm_backbone import LLMBackbone
from searchagent.agent.persona import Persona, PersonaGenerator, TRECPersonaExtractor
from searchagent.agent.search_agent import SearchUserAgent
from searchagent.agent.prompts import MACRO_ACTIONS
from searchagent.config import SearchAgentConfig
from searchagent.environment.search_env import SearchEnvironment, ReplayEnvironment, SessionRecord
from searchagent.environment.serp import TRECSERPLoader, TRECSession, YandexSERPLoader, OrcasSERPLoader
from searchagent.evaluation.metrics import composite_fidelity_score, replay_alignment_metrics
from searchagent.evaluation.reference_data import compute_reference_from_sessions
from searchagent.evaluation.statistical_tests import (
    bootstrap_ci,
    paired_permutation_test,
    cohens_d,
    significance_marker,
    pairwise_significance_table,
    format_significance_table_latex,
)
from searchagent.utils.logging import setup_logging

logger = logging.getLogger(__name__)

PAPER_BASELINES = ["react", "polyskill", "reflexion", "cot", "tot", "usersimcrs_v2", "cshi", "usimagent", "bases", "agent4rec", "simuser", "convsearch_r1"]
DIAGNOSTIC_AGENTS = ["random", "rule_based", "naive_llm"]
AGENT_TYPES = ["full", "full_pp"] + PAPER_BASELINES + DIAGNOSTIC_AGENTS

# SearchAgent++ : loaded once in main() from --calibrator; used by full_pp agent.
_CALIBRATOR = None


def create_agent(
    agent_type: str,
    persona: Persona,
    llm: LLMBackbone,
    config: SearchAgentConfig,
    calibrator=None,
):
    if agent_type in ("full", "full_pp"):
        return SearchUserAgent(persona=persona, llm=llm, config=config,
                               calibrator=calibrator if calibrator is not None else _CALIBRATOR)
    if agent_type == "random":
        return RandomAgent(persona=persona, config=config)
    if agent_type == "rule_based":
        return RuleBasedAgent(persona=persona, config=config)
    if agent_type == "naive_llm":
        return NaiveLLMAgent(persona=persona, llm=llm, config=config)
    paper = {
        "react": ReActSearchAgent,
        "polyskill": PolySkillAgent,
        "reflexion": ReflexionAgent,
        "cot": CoTAgent,
        "tot": TreeOfThoughtsAgent,
        "usersimcrs_v2": UserSimCRSv2Agent,
        "cshi": CSHIAgent,
        "usimagent": USimAgent,
        "bases": BASESAgent,
        "agent4rec": Agent4RecAgent,
        "simuser": SimUSERAgent,
        "convsearch_r1": ConvSearchR1Agent,
    }
    if agent_type in paper:
        return paper[agent_type](persona=persona, llm=llm, config=config)
    raise ValueError(f"Unknown agent type: {agent_type}")


def simulate_session(agent, env: SearchEnvironment, persona: Persona, max_steps: int, timeout_prob: float = 0.0, pre_click_abandon_prob: float = 0.0) -> SessionRecord:
    serp = env.reset(persona.persona_id, persona.search_topic)
    done = False
    while not done and agent.step_count < max_steps:
        serp_context = serp.to_prompt_dict() if serp else {"query": agent.current_query, "page": 1, "results": [], "text_blind": True}
        action, _ = agent.decide_action(serp_context)
        click_index = None
        new_query = None
        satisfaction = None
        dwell_time = None
        if action in ("click_organic", "click_ad", "click_result"):
            click_index = agent.select_click_target(serp_context, action, serp_results=serp.results if serp else [])
            if serp and 0 <= click_index < len(serp.results):
                result = serp.results[click_index]
                if hasattr(agent, "rate_satisfaction"):
                    satisfaction = agent.rate_satisfaction(result.title, result.snippet, result.is_ad, result.url)
                    dwell_time = agent.estimate_dwell_time(satisfaction, result.is_ad)
                else:
                    satisfaction = 3.0
                    dwell_time = 35.0
                if isinstance(agent, SearchUserAgent):
                    agent.record_action(action=action, target=result.url, result_title=result.title, is_ad=result.is_ad, satisfaction=satisfaction, dwell_time=dwell_time)
                else:
                    agent.record_action(action=action, target=result.url, result_title=result.title, is_ad=result.is_ad, satisfaction=satisfaction)
            else:
                agent.record_action(action=action)
        elif action in ("reformulate", "new_query"):
            new_query = agent.generate_query(is_reformulation=action == "reformulate")
            agent.record_action(action=action, target=new_query)
            agent.current_query = new_query
        else:
            agent.record_action(action=action)
        serp, done = env.step(action=action, click_index=click_index, new_query=new_query)
        if satisfaction is not None:
            env.update_last_click(satisfaction, dwell_time)
    return env.get_session_record()


def run_single_config(
    agent_type: str,
    config: SearchAgentConfig,
    llm: LLMBackbone,
    personas: List[Persona],
    seed: int,
) -> Tuple[List[SessionRecord], Dict[str, Any]]:
    """Run all sessions for one agent_type + seed combination."""
    np.random.seed(seed)
    env = SearchEnvironment(config.environment, seed=seed)
    results = []

    for persona in personas:
        agent = create_agent(agent_type, persona, llm, config)
        try:
            record = simulate_session(
                agent, env, persona, config.simulation.max_session_steps,
            )
            results.append(record)
        except Exception as e:
            logger.error("Session %d failed for %s: %s", persona.persona_id, agent_type, e)

    metrics = composite_fidelity_score(results)
    metrics["agent_type"] = agent_type
    metrics["seed"] = seed
    return results, metrics


def extract_trait_distribution(trec_sessions: List[TRECSession]) -> Dict[str, List[str]]:
    """Extract trait value distributions from a set of TREC sessions.

    Returns dict mapping trait_name -> list of trait values (one per session).
    Used for cross-dataset persona transfer.
    """
    dist: Dict[str, List[str]] = {
        "patience_level": [], "search_expertise": [],
        "decision_style": [], "ad_sensitivity": [],
    }
    for s in trec_sessions:
        traits = TRECPersonaExtractor.extract_from_session(s)
        for k in dist:
            dist[k].append(traits[k])
    return dist


def sample_persona_from_distribution(
    trait_dist: Dict[str, List[str]],
    trec_session: TRECSession,
    persona_id: int,
    rng: np.random.RandomState,
) -> Persona:
    """Create a persona by sampling traits from a source distribution.

    The topic/query comes from the target session; traits from the source.
    """
    from searchagent.agent.persona import Persona
    traits = {}
    for k, vals in trait_dist.items():
        traits[k] = rng.choice(vals)
    return Persona(
        persona_id=persona_id,
        age=30,
        search_expertise=traits["search_expertise"],
        ad_sensitivity=traits["ad_sensitivity"],
        decision_style=traits["decision_style"],
        patience_level=traits["patience_level"],
        search_topic=trec_session.interactions[0].query if trec_session.interactions else "",
        search_goal=trec_session.topic_desc[:200] if trec_session.topic_desc else "",
    )


def run_replay_config(
    agent_type: str,
    config: SearchAgentConfig,
    llm: LLMBackbone,
    trec_sessions: List[TRECSession],
    seed: int,
    persona_trait_dist: Optional[Dict[str, List[str]]] = None,
    serp_perturbation: str = 'none',
    dataset_reference=None,
) -> Tuple[List[SessionRecord], Dict[str, Any]]:
    """Run agent on real TREC SERPs and compare with ground truth.

    For each TREC session, the agent sees the same SERP the real user saw.
    We compute per-session alignment: click accuracy, action agreement, etc.

    If persona_trait_dist is provided, personas are sampled from that
    distribution (cross-dataset transfer) instead of per-session extraction.

    If dataset_reference is provided, PFD is computed against those stats
    instead of the default TREC 2014 reference.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    results = []
    click_hits = 0   # agent clicked same rank as real user
    click_total = 0  # total real user clicks
    action_agree = 0  # action type agreement
    action_total = 0
    agent_clicks_total = 0

    for trec_s in trec_sessions:
        # Create persona: cross-dataset transfer or native per-session
        if persona_trait_dist is not None:
            persona = sample_persona_from_distribution(
                persona_trait_dist, trec_s, persona_id=trec_s.session_num, rng=rng)
        else:
            persona = TRECPersonaExtractor.create_persona(trec_s, persona_id=trec_s.session_num)

        agent = create_agent(agent_type, persona, llm, config)
        env = ReplayEnvironment(trec_s, config.environment,
                                serp_perturbation=serp_perturbation, rng=rng)
        try:
            record = simulate_session(
                agent, env, persona, config.simulation.max_session_steps,
            )
            results.append(record)

            # Per-interaction click comparison (multiset, not set)
            # Compare clicks within each interaction to preserve repeated ranks
            agent_click_ranks_list = [c.rank for c in record.clicks]
            real_click_ranks_list = trec_s.click_positions
            agent_clicks_total += len(agent_click_ranks_list)
            click_total += len(real_click_ranks_list)

            # Multiset intersection: count matches up to multiplicity
            agent_counter = Counter(agent_click_ranks_list)
            real_counter = Counter(real_click_ranks_list)
            for rank, count in agent_counter.items():
                click_hits += min(count, real_counter.get(rank, 0))

            # Action agreement: did agent abandon when user abandoned?
            real_abandoned = not trec_s.has_clicks
            action_total += 1
            if record.abandoned == real_abandoned:
                action_agree += 1

        except Exception as e:
            logger.error("Replay session %d failed for %s: %s",
                         trec_s.session_num, agent_type, e)

    metrics = composite_fidelity_score(results, reference=dataset_reference)
    metrics["agent_type"] = agent_type
    metrics["seed"] = seed
    metrics["replay_sessions"] = len(trec_sessions)
    replay = replay_alignment_metrics(results, trec_sessions)
    metrics.update(replay)
    metrics["click_precision"] = metrics["replay_precision"]
    metrics["click_recall"] = metrics["replay_recall"]
    metrics["click_f1"] = metrics["replay_f1"]
    metrics["abandon_agreement"] = metrics["abandonment_agreement"]

    return results, metrics


def aggregate_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across multiple seeds."""
    if not all_metrics:
        return {}

    numeric_keys = [k for k, v in all_metrics[0].items() if isinstance(v, (int, float))]
    agg = {}
    for key in numeric_keys:
        values = [m[key] for m in all_metrics if key in m and np.isfinite(m[key])]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))

    agg["n_seeds"] = len(all_metrics)
    agg["agent_type"] = all_metrics[0].get("agent_type", "unknown")
    return agg


def per_trait_analysis(
    sessions: List[SessionRecord],
    personas: List[Persona],
) -> Dict[str, Dict[str, float]]:
    """Analyze behavioral patterns grouped by persona traits."""
    trait_groups = {
        "ad_sensitivity": {},
        "patience_level": {},
        "decision_style": {},
        "search_expertise": {},
    }

    persona_map = {p.persona_id: p for p in personas}

    for trait_name in trait_groups:
        groups = {}
        for session in sessions:
            persona = persona_map.get(session.persona_id)
            if persona is None:
                continue
            trait_val = getattr(persona, trait_name, "unknown")
            if trait_val not in groups:
                groups[trait_val] = []
            groups[trait_val].append(session)

        for trait_val, group_sessions in groups.items():
            if not group_sessions:
                continue
            n = len(group_sessions)
            avg_clicks = np.mean([len(s.clicks) for s in group_sessions])
            avg_ad_clicks = np.mean([s.ad_clicks for s in group_sessions])
            avg_length = np.mean([s.session_length for s in group_sessions])
            abandon_rate = sum(1 for s in group_sessions if s.abandoned) / n
            reform_rate = sum(1 for s in group_sessions if s.num_queries > 1) / n

            trait_groups[trait_name][trait_val] = {
                "n_sessions": n,
                "avg_clicks": round(avg_clicks, 2),
                "avg_ad_clicks": round(avg_ad_clicks, 2),
                "avg_session_length": round(avg_length, 2),
                "abandon_rate": round(abandon_rate, 3),
                "reformulation_rate": round(reform_rate, 3),
            }

    return trait_groups


def main():
    parser = argparse.ArgumentParser(description="Run SearchAgent experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--agents", nargs="+", default=["full"],
        help=f"Agent types to run. Available: {AGENT_TYPES}",
    )
    parser.add_argument("--num_sessions", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output_dir", type=str, default="output/experiments")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["synthetic", "replay"],
                        help="synthetic: generated SERPs; replay: real TREC SERPs")
    parser.add_argument("--trec_data", type=str,
                        default="data/search_logs/trec_session2014.xml",
                        help="Path to TREC Session Track XML (for replay mode)")
    parser.add_argument("--persona_source", type=str, nargs="+", default=None,
                        help="TREC XML(s) to extract persona trait distribution from "
                             "(cross-dataset transfer). If not set, per-session extraction.")
    parser.add_argument("--serp_perturbation", type=str, default="none",
                        choices=["none", "shuffle", "reverse"],
                        help="SERP result perturbation for ranking quality sensitivity")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip (agent, seed) pairs whose metrics.json already exists")
    parser.add_argument("--calibrator", type=str, default=None,
                        help="Path to calibrator.pt; overrides config.calibration.calibrator_path")
    parser.add_argument("--ablate", type=str, default=None,
                        choices=["no_f", "no_gate", "no_click"],
                        help="Disable one learned component at inference (ablation study)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = SearchAgentConfig.from_yaml(args.config)

    if args.ablate:
        import searchagent.agent.calibration as _cal
        _cal.ABLATE = args.ablate
        logging.getLogger(__name__).info("ABLATION active: %s", args.ablate)

    # SearchAgent++: calibrator_path from CLI overrides yaml; yaml is the default.
    calib_path = args.calibrator or config.calibration.calibrator_path
    if calib_path and Path(calib_path).exists():
        import torch
        from searchagent.agent.calibration import Calibrator
        global _CALIBRATOR
        ckpt = torch.load(calib_path, map_location="cpu")
        # Prefer dims baked into the checkpoint (the shape the weights were
        # trained at); fall back to the yaml config for legacy checkpoints.
        dims = ckpt.get("dims", {})
        _CALIBRATOR = Calibrator(
            n_intent=dims.get("n_intent", len(config.intent.states)),
            hidden1=dims.get("hidden1", dims.get("hidden", config.calibration.hidden1)),
            hidden2=dims.get("hidden2", config.calibration.hidden2),
            dropout=dims.get("dropout", config.calibration.dropout),
            alpha_max=dims.get("alpha_max", config.calibration.alpha_max),
            exam_dim=dims.get("exam_dim", config.calibration.exam_dim),
            attr_dim=dims.get("attr_dim", config.calibration.attr_dim),
            epsilon=dims.get("epsilon", config.calibration.epsilon))
        # strict=False: legacy checkpoints contain belief.* keys (the removed
        # dead BeliefUpdater); ignore them so existing calibrators still load.
        _CALIBRATOR.load_state_dict(ckpt["state_dict"], strict=False)
        _CALIBRATOR.eval()
        logger.info("Loaded SearchAgent++ calibrator from %s", calib_path)
    else:
        logger.warning("Calibrator checkpoint not found; SearchAgent (full/full_pp) will use the structured LLM prior.")

    if args.num_sessions:
        config.simulation.num_sessions = args.num_sessions
    num_sessions = config.simulation.num_sessions

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load TREC/Yandex data for replay mode
    trec_sessions = None
    persona_trait_dist = None
    if args.mode == "replay":
        # Reference pool: load a large sample (or full TREC xml) so the PFD
        # reference is the dataset's stable real distribution, NOT the 20-session
        # eval sample (which is noisy and self-referential). For the huge gz logs
        # we cap the pool at 500 sessions, which is plenty for stable stats.
        REF_POOL = 500
        if "orcas" in args.trec_data.lower():
            loader = OrcasSERPLoader(args.trec_data, max_sessions=REF_POOL)
            logger.info("Replay mode: loaded ORCAS sessions from %s", args.trec_data)
        elif args.trec_data.endswith(".gz"):
            loader = YandexSERPLoader(args.trec_data, max_sessions=REF_POOL)
            logger.info("Replay mode: loaded Yandex sessions from %s", args.trec_data)
        else:
            loader = TRECSERPLoader(args.trec_data)
            logger.info("Replay mode: loaded TREC sessions from %s", args.trec_data)

        ref_pool_sessions = loader.sessions

        # Compute per-dataset reference statistics from the FULL pool (before
        # truncating to the eval sample) so the PFD target is fixed & reproducible.
        dataset_reference = compute_reference_from_sessions(ref_pool_sessions)
        logger.info("Per-dataset reference (from %d-session pool): "
                    "clicks/session=%.2f, abandon=%.3f, reform=%.3f",
                    len(ref_pool_sessions),
                    dataset_reference.clicks_per_session, dataset_reference.abandon_rate,
                    dataset_reference.reformulation_rate)

        # Now truncate to the eval sample.
        trec_sessions = ref_pool_sessions
        if args.num_sessions and args.num_sessions < len(trec_sessions):
            trec_sessions = trec_sessions[:args.num_sessions]
        num_sessions = len(trec_sessions)
        logger.info("Replay mode: %d sessions used for evaluation", num_sessions)

        # Cross-dataset persona transfer: extract traits from source dataset(s)
        if args.persona_source:
            source_sessions = []
            for src_path in args.persona_source:
                src_loader = TRECSERPLoader(src_path)
                source_sessions.extend(src_loader.sessions)
                logger.info("Persona source: %d sessions from %s", len(src_loader.sessions), src_path)
            persona_trait_dist = extract_trait_distribution(source_sessions)
            logger.info("Cross-dataset transfer: %d source sessions → trait distribution", len(source_sessions))

    # Generate consistent personas across all agents/seeds (synthetic mode)
    personas = None
    if args.mode == "synthetic":
        gen = PersonaGenerator(config.persona, seed=42)
        personas = gen.generate(num_sessions)

    # Load LLM once (shared across all LLM-using agents)
    needs_llm = any(a not in ("random", "rule_based") for a in args.agents)
    llm = LLMBackbone(config.llm) if needs_llm else None
    if llm is not None:
        llm.load()

    all_results = {}
    summary_table = []

    for agent_type in args.agents:
        if agent_type not in AGENT_TYPES:
            logger.warning("Skipping unknown agent type: %s", agent_type)
            continue

        logger.info("=" * 60)
        logger.info("Running agent: %s (%s mode)", agent_type, args.mode)

        seed_metrics = []
        seed_sessions = []
        t0 = time.time()

        for seed in args.seeds:
            seed_dir = output_dir / agent_type / f"seed_{seed}"
            metrics_path = seed_dir / "metrics.json"

            if args.skip_existing and metrics_path.exists():
                logger.info("  Seed %d: skipping (metrics.json exists)", seed)
                with open(metrics_path) as f:
                    seed_metrics.append(json.load(f))
                continue

            logger.info("  Seed %d: %d sessions", seed, num_sessions)
            if args.mode == "replay":
                sessions, metrics = run_replay_config(
                    agent_type, config, llm, trec_sessions, seed,
                    persona_trait_dist=persona_trait_dist,
                    serp_perturbation=args.serp_perturbation,
                    dataset_reference=dataset_reference,
                )
            else:
                sessions, metrics = run_single_config(
                    agent_type, config, llm, personas, seed,
                )
            seed_metrics.append(metrics)
            # Trait analysis is synthetic-mode only; keeping replay sessions in RAM
            # across all seeds leaks ~hundreds of MB per run and triggers OOM.
            if personas:
                seed_sessions.extend(sessions)

            seed_dir.mkdir(parents=True, exist_ok=True)
            with open(seed_dir / "sessions.json", "w") as f:
                json.dump([s.to_dict() for s in sessions], f, indent=2, ensure_ascii=False)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            del sessions
            gc.collect()

        elapsed = time.time() - t0
        agg = aggregate_metrics(seed_metrics)
        agg["total_time_s"] = round(elapsed, 1)
        agg["time_per_session_s"] = round(
            elapsed / max(len(seed_sessions) or num_sessions * len(args.seeds), 1), 2
        )

        # Per-trait analysis (only available in synthetic mode with personas)
        if personas:
            trait_analysis = per_trait_analysis(seed_sessions, personas)
            agg["trait_analysis"] = trait_analysis

        all_results[agent_type] = agg
        summary_table.append(agg)

        logger.info("  %s: PFD_mean=%.4f (±%.4f), time=%.1fs",
                     agent_type,
                     agg.get("PFD_mean", float("inf")),
                     agg.get("PFD_std", 0),
                     elapsed)

    # ── Statistical significance testing ────────────────────────────────
    # Collect per-seed metric dicts keyed by agent for pairwise tests
    per_seed_by_agent = {}
    for agent_type in args.agents:
        if agent_type not in all_results:
            continue
        seed_dir_base = output_dir / agent_type
        per_seed_by_agent[agent_type] = []
        for seed in args.seeds:
            mpath = seed_dir_base / f"seed_{seed}" / "metrics.json"
            if mpath.exists():
                with open(mpath) as f:
                    per_seed_by_agent[agent_type].append(json.load(f))

    # Bootstrap CIs for each agent's PFD
    for agent_type, seed_dicts in per_seed_by_agent.items():
        pfd_vals = np.array([d.get("PFD", float("nan")) for d in seed_dicts])
        pfd_vals = pfd_vals[np.isfinite(pfd_vals)]
        if len(pfd_vals) >= 2:
            mean, ci_lo, ci_hi = bootstrap_ci(pfd_vals)
            all_results[agent_type]["PFD_ci_lower"] = round(ci_lo, 4)
            all_results[agent_type]["PFD_ci_upper"] = round(ci_hi, 4)

    # Pairwise significance (all pairs, on PFD)
    sig_results = []
    if len(per_seed_by_agent) >= 2 and all(
        len(v) >= 2 for v in per_seed_by_agent.values()
    ):
        sig_results = pairwise_significance_table(
            per_seed_by_agent, metric="PFD", alpha=0.05,
        )
        # Save significance table as LaTeX
        sig_latex = format_significance_table_latex(sig_results)
        (output_dir / "significance_tests.tex").write_text(sig_latex)

        # Also save as JSON
        sig_json = [
            {
                "agent_a": r.agent_a, "agent_b": r.agent_b,
                "mean_a": r.mean_a, "mean_b": r.mean_b,
                "diff": round(r.diff, 4), "p_value": round(r.p_value, 4),
                "effect_size": round(r.effect_size, 2),
                "significant": r.significant,
            }
            for r in sig_results
        ]
        with open(output_dir / "significance_tests.json", "w") as f:
            json.dump(sig_json, f, indent=2)

    # Save overall summary
    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    header = f"{'Agent':<20} {'PFD':>8} {'±':>6} {'95% CI':>15} {'ClickJSD':>9} {'Clk/S':>8} {'Time':>8}"
    logger.info(header)
    logger.info("-" * 80)

    for agg in summary_table:
        agent = agg.get("agent_type", "?")
        ci_lo = agg.get("PFD_ci_lower", 0)
        ci_hi = agg.get("PFD_ci_upper", 0)
        ci_str = f"[{ci_lo:.4f},{ci_hi:.4f}]" if ci_lo or ci_hi else "       --      "
        logger.info(
            f"{agent:<20} "
            f"{agg.get('PFD_mean', 0):.4f} "
            f"{agg.get('PFD_std', 0):.4f} "
            f"{ci_str:>15} "
            f"{agg.get('click_position_jsd_mean', 0):.4f}   "
            f"{agg.get('clicks_per_session_mean', 0):.4f} "
            f"{agg.get('total_time_s', 0):>6.1f}s"
        )

    # Print significance results
    if sig_results:
        logger.info("")
        logger.info("PAIRWISE SIGNIFICANCE TESTS (PFD, permutation test)")
        logger.info("-" * 80)
        logger.info(f"{'Agent A':<20} {'Agent B':<20} {'Δ':>8} {'p':>8} {'d':>6} {'Sig':>5}")
        logger.info("-" * 80)
        for r in sig_results:
            marker = significance_marker(r.p_value)
            logger.info(
                f"{r.agent_a:<20} {r.agent_b:<20} "
                f"{r.diff:+.4f} {r.p_value:.4f} {r.effect_size:>5.2f} {marker:>5}"
            )

    logger.info("=" * 80)
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
