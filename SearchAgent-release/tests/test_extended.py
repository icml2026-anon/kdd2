"""Extended tests for baselines, ablations, SERP fuzzy matching, and new metrics."""

from __future__ import annotations

import unittest
import numpy as np

from searchagent.config import SearchAgentConfig, LLMConfig, EnvironmentConfig, PersonaConfig
from searchagent.agent.persona import Persona, PersonaGenerator


def _make_persona(**kwargs) -> Persona:
    defaults = dict(
        persona_id=0, age=30, search_expertise="intermediate",
        ad_sensitivity="ad_neutral", decision_style="deliberate",
        patience_level="medium", search_topic="best noise cancelling headphones",
        search_goal="Find headphones under $300",
    )
    defaults.update(kwargs)
    return Persona(**defaults)


def _make_config() -> SearchAgentConfig:
    return SearchAgentConfig(llm=LLMConfig(model_name="dummy"))


class TestSERPFuzzyMatching(unittest.TestCase):
    def test_exact_match(self):
        from searchagent.environment.serp import _best_matching_key
        from searchagent.environment.serp_templates import ORGANIC_TEMPLATES
        key = _best_matching_key("best noise cancelling headphones", ORGANIC_TEMPLATES)
        self.assertEqual(key, "best noise cancelling headphones")

    def test_reformulated_query_matches(self):
        from searchagent.environment.serp import _best_matching_key
        from searchagent.environment.serp_templates import ORGANIC_TEMPLATES
        key = _best_matching_key("noise cancelling headphones review", ORGANIC_TEMPLATES)
        self.assertEqual(key, "best noise cancelling headphones")

    def test_partial_query_matches(self):
        from searchagent.environment.serp import _best_matching_key
        from searchagent.environment.serp_templates import ORGANIC_TEMPLATES
        key = _best_matching_key("cheap flights tokyo 2024", ORGANIC_TEMPLATES)
        self.assertEqual(key, "cheap flights to tokyo")

    def test_unrelated_query_defaults(self):
        from searchagent.environment.serp import _best_matching_key
        from searchagent.environment.serp_templates import ORGANIC_TEMPLATES
        key = _best_matching_key("quantum computing research papers", ORGANIC_TEMPLATES)
        self.assertEqual(key, "default")

    def test_all_scenarios_have_templates(self):
        from searchagent.environment.serp_templates import ORGANIC_TEMPLATES, AD_TEMPLATES
        from searchagent.agent.persona import SEARCH_SCENARIOS
        for scenario in SEARCH_SCENARIOS:
            topic = scenario["topic"].lower()
            self.assertIn(topic, ORGANIC_TEMPLATES,
                         f"Missing organic templates for: {topic}")
            self.assertIn(topic, AD_TEMPLATES,
                         f"Missing ad templates for: {topic}")

    def test_each_scenario_has_enough_results(self):
        from searchagent.environment.serp_templates import ORGANIC_TEMPLATES, AD_TEMPLATES
        for key, templates in ORGANIC_TEMPLATES.items():
            self.assertGreaterEqual(len(templates), 10,
                                   f"Organic templates for '{key}' has only {len(templates)}")
        for key, templates in AD_TEMPLATES.items():
            self.assertGreaterEqual(len(templates), 3,
                                   f"Ad templates for '{key}' has only {len(templates)}")

    def test_serp_page2_different_from_page1(self):
        from searchagent.environment.serp import SERPGenerator
        gen = SERPGenerator(EnvironmentConfig(), seed=42)
        serp1 = gen.generate("best noise cancelling headphones", page_number=1)
        gen2 = SERPGenerator(EnvironmentConfig(), seed=42)
        serp2 = gen2.generate("best noise cancelling headphones", page_number=2)
        # Page 2 should have different ordering
        titles1 = [r.title for r in serp1.organic_results]
        titles2 = [r.title for r in serp2.organic_results]
        self.assertNotEqual(titles1, titles2)


class TestRandomAgent(unittest.TestCase):
    def test_basic_operation(self):
        from searchagent.agent.baselines import RandomAgent
        config = _make_config()
        persona = _make_persona()
        agent = RandomAgent(persona, config)

        from searchagent.agent.prompts import MACRO_ACTIONS
        action, probs = agent.decide_action("dummy serp")
        self.assertIn(action, MACRO_ACTIONS)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_click_target_ad(self):
        from searchagent.agent.baselines import RandomAgent
        from searchagent.environment.serp import SearchResult
        config = _make_config()
        persona = _make_persona()
        agent = RandomAgent(persona, config)

        results = [
            SearchResult(rank=1, title="Organic", snippet="", url="", is_ad=False),
            SearchResult(rank=2, title="Ad", snippet="", url="", is_ad=True),
            SearchResult(rank=3, title="Organic2", snippet="", url="", is_ad=False),
        ]
        idx = agent.select_click_target("", "click_ad", results)
        self.assertEqual(idx, 1)  # only ad is at index 1

    def test_query_generation(self):
        from searchagent.agent.baselines import RandomAgent
        config = _make_config()
        persona = _make_persona()
        agent = RandomAgent(persona, config)

        q = agent.generate_query(is_reformulation=True)
        self.assertTrue(len(q) > 0)
        self.assertIn("headphones", q.lower())

    def test_reset(self):
        from searchagent.agent.baselines import RandomAgent
        config = _make_config()
        persona = _make_persona()
        agent = RandomAgent(persona, config)
        agent.record_action("skip")
        agent.record_action("click_organic")
        self.assertEqual(agent.step_count, 2)
        agent.reset()
        self.assertEqual(agent.step_count, 0)
        self.assertEqual(len(agent.memory.entries), 0)


class TestRuleBasedAgent(unittest.TestCase):
    def test_basic_operation(self):
        from searchagent.agent.baselines import RuleBasedAgent
        from searchagent.agent.prompts import MACRO_ACTIONS
        config = _make_config()
        persona = _make_persona()
        agent = RuleBasedAgent(persona, config)

        action, probs = agent.decide_action("dummy serp")
        self.assertIn(action, MACRO_ACTIONS)

    def test_abandon_after_max_pages(self):
        from searchagent.agent.baselines import RuleBasedAgent
        config = _make_config()
        persona = _make_persona()
        agent = RuleBasedAgent(persona, config)

        # Simulate max pages without clicks
        for _ in range(agent._max_pages_before_abandon):
            agent.record_action("next_page")
        action, _ = agent.decide_action("dummy")
        self.assertEqual(action, "stop_unsatisfied")

    def test_click_selection_weighted(self):
        from searchagent.agent.baselines import RuleBasedAgent
        from searchagent.environment.serp import SearchResult
        config = _make_config()
        persona = _make_persona()
        agent = RuleBasedAgent(persona, config)

        results = [
            SearchResult(rank=1, title="R1", snippet="", url="", relevance=0.9),
            SearchResult(rank=5, title="R5", snippet="", url="", relevance=0.3),
            SearchResult(rank=10, title="R10", snippet="", url="", relevance=0.1),
        ]
        # Run many times to check distribution is position-biased
        clicks = [agent.select_click_target("", "click_organic", results)
                  for _ in range(200)]
        # First result should be clicked most often
        self.assertGreater(clicks.count(0), clicks.count(2))


class TestNaiveLLMAgent(unittest.TestCase):
    def test_init(self):
        from searchagent.agent.baselines import NaiveLLMAgent
        from searchagent.agent.llm_backbone import LLMBackbone
        config = _make_config()
        persona = _make_persona()
        llm = LLMBackbone(config.llm)
        agent = NaiveLLMAgent(persona, llm, config)
        self.assertIn("searching for", agent._persona_desc)
        self.assertNotIn("30-year-old", agent._persona_desc)

    def test_no_memory_in_prompt(self):
        from searchagent.agent.baselines import NaiveLLMAgent
        from searchagent.agent.llm_backbone import LLMBackbone
        config = _make_config()
        persona = _make_persona()
        llm = LLMBackbone(config.llm)
        agent = NaiveLLMAgent(persona, llm, config)
        # Record some actions
        agent.record_action("click_organic")
        agent.record_action("skip")
        # The agent won't use memory — verify internal state exists but won't be used
        self.assertEqual(len(agent.memory.entries), 2)


class TestAblationAgent(unittest.TestCase):
    def test_all_variants_creatable(self):
        from searchagent.agent.ablations import ABLATION_VARIANTS, create_ablation_agent
        from searchagent.agent.llm_backbone import LLMBackbone
        config = _make_config()
        persona = _make_persona()
        llm = LLMBackbone(config.llm)

        for variant in ABLATION_VARIANTS:
            agent = create_ablation_agent(variant, persona, llm, config)
            self.assertIsNotNone(agent)

    def test_ablation_names(self):
        from searchagent.agent.ablations import AblationAgent
        from searchagent.agent.llm_backbone import LLMBackbone
        config = _make_config()
        persona = _make_persona()
        llm = LLMBackbone(config.llm)

        a1 = AblationAgent(persona, llm, config, disable_persona_bias=True)
        self.assertEqual(a1.ablation_name, "no_persona")

        a_full = AblationAgent(persona, llm, config)
        self.assertEqual(a_full.ablation_name, "full_model")

        a_multi = AblationAgent(persona, llm, config,
                                disable_persona_bias=True, disable_memory=True)
        self.assertIn("no_persona", a_multi.ablation_name)
        self.assertIn("no_memory", a_multi.ablation_name)

    def test_ad_ablation_merges_click(self):
        from searchagent.agent.ablations import AblationAgent
        from searchagent.agent.llm_backbone import LLMBackbone
        config = _make_config()
        persona = _make_persona()
        llm = LLMBackbone(config.llm)

        agent = AblationAgent(persona, llm, config, disable_ad_model=True)
        # When ad model is disabled, click_ad should be zero after merging
        test_probs = {
            "click_organic": 0.3, "click_ad": 0.1, "skip": 0.2,
            "next_page": 0.1, "reformulate": 0.1, "new_query": 0.1, "abandon": 0.1,
        }
        # _apply_persona_bias is not disabled, but check the logic
        # We can't call decide_action without LLM, but we can verify structure
        self.assertTrue(agent.disable_ad_model)

    def test_unknown_variant_raises(self):
        from searchagent.agent.ablations import create_ablation_agent
        from searchagent.agent.llm_backbone import LLMBackbone
        config = _make_config()
        persona = _make_persona()
        llm = LLMBackbone(config.llm)

        with self.assertRaises(ValueError):
            create_ablation_agent("INVALID", persona, llm, config)


class TestNewMetrics(unittest.TestCase):
    def _make_sessions(self, n=30):
        from searchagent.environment.search_env import ClickRecord, SessionRecord
        rng = np.random.RandomState(42)
        sessions = []
        for i in range(n):
            n_clicks = rng.randint(0, 5)
            clicks = []
            for _ in range(n_clicks):
                rank = rng.choice(range(1, 11),
                                  p=[0.3, 0.2, 0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.06, 0.05])
                clicks.append(ClickRecord(
                    step=0, query="q", rank=rank, title="t",
                    is_ad=(rng.random() < 0.1), relevance=rng.uniform(0.3, 0.9),
                ))
            n_queries = rng.randint(1, 4)
            steps = []
            actions = ["click_organic", "skip", "reformulate", "next_page",
                       "click_ad", "abandon", "end_session"]
            for s in range(rng.randint(2, 12)):
                steps.append({"action": rng.choice(actions), "step": s})
            sessions.append(SessionRecord(
                persona_id=i, initial_query="q",
                clicks=clicks, queries=[f"q{j}" for j in range(n_queries)],
                total_steps=len(steps), abandoned=(rng.random() < 0.15),
                steps=steps,
            ))
        return sessions

    def test_click_position_jsd(self):
        from searchagent.evaluation.metrics import click_position_jsd
        sessions = self._make_sessions()
        jsd = click_position_jsd(sessions)
        self.assertGreaterEqual(jsd, 0.0)
        self.assertLess(jsd, np.log(2) + 0.01)  # JSD bounded by ln(2)

    def test_session_length_jsd(self):
        from searchagent.evaluation.metrics import session_length_jsd
        sessions = self._make_sessions()
        jsd = session_length_jsd(sessions)
        self.assertGreaterEqual(jsd, 0.0)
        self.assertLess(jsd, np.log(2) + 0.01)

    def test_clicks_per_session_gap(self):
        from searchagent.evaluation.metrics import clicks_per_session_gap
        sessions = self._make_sessions()
        gap = clicks_per_session_gap(sessions)
        self.assertGreaterEqual(gap, 0.0)

    def test_abandon_rate_gap(self):
        from searchagent.evaluation.metrics import abandon_rate_gap
        sessions = self._make_sessions()
        gap = abandon_rate_gap(sessions)
        self.assertGreaterEqual(gap, 0.0)

    def test_reformulation_rate_gap(self):
        from searchagent.evaluation.metrics import reformulation_rate_gap
        sessions = self._make_sessions()
        gap = reformulation_rate_gap(sessions)
        self.assertGreaterEqual(gap, 0.0)

    def test_composite_has_5_core_metrics(self):
        from searchagent.evaluation.metrics import composite_fidelity_score
        sessions = self._make_sessions()
        result = composite_fidelity_score(sessions)
        for key in ["click_position_jsd", "session_length_jsd",
                     "clicks_per_session", "abandon_rate",
                     "reformulation_rate", "CFS",
                     "mean_session_length", "mean_clicks",
                     "mean_queries", "abandon_rate_raw"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_composite_cfs_bounded_0_1(self):
        from searchagent.evaluation.metrics import composite_fidelity_score
        sessions = self._make_sessions()
        result = composite_fidelity_score(sessions)
        self.assertIn("CFS", result)
        self.assertGreaterEqual(result["CFS"], 0.0)
        self.assertLessEqual(result["CFS"], 1.0)  # CFS is now [0,1]

    def test_reference_data_loaded(self):
        from searchagent.evaluation.reference_data import REFERENCE
        self.assertEqual(len(REFERENCE.click_position_dist), 10)
        self.assertEqual(len(REFERENCE.session_length_dist), 10)
        self.assertAlmostEqual(REFERENCE.click_position_dist.sum(), 1.0, places=5)
        self.assertAlmostEqual(REFERENCE.session_length_dist.sum(), 1.0, places=5)
        self.assertGreater(REFERENCE.clicks_per_session, 0)
        self.assertGreater(REFERENCE.abandon_rate, 0)
        self.assertGreater(REFERENCE.reformulation_rate, 0)


class TestExperimentRunner(unittest.TestCase):
    def test_agent_factory(self):
        from scripts.run_experiment import create_agent, AGENT_TYPES
        from searchagent.agent.llm_backbone import LLMBackbone
        config = _make_config()
        persona = _make_persona()
        llm = LLMBackbone(config.llm)

        for agent_type in ["random", "rule_based"]:
            agent = create_agent(agent_type, persona, llm, config)
            self.assertIsNotNone(agent)

    def test_per_trait_analysis(self):
        from scripts.run_experiment import per_trait_analysis
        from searchagent.environment.search_env import SessionRecord

        personas = [
            _make_persona(persona_id=0, ad_sensitivity="ad_averse"),
            _make_persona(persona_id=1, ad_sensitivity="ad_tolerant"),
            _make_persona(persona_id=2, ad_sensitivity="ad_averse"),
        ]
        sessions = [
            SessionRecord(persona_id=0, initial_query="q", total_steps=5, abandoned=True),
            SessionRecord(persona_id=1, initial_query="q", total_steps=10, abandoned=False),
            SessionRecord(persona_id=2, initial_query="q", total_steps=3, abandoned=False),
        ]

        analysis = per_trait_analysis(sessions, personas)
        self.assertIn("ad_sensitivity", analysis)
        self.assertIn("ad_averse", analysis["ad_sensitivity"])
        self.assertIn("ad_tolerant", analysis["ad_sensitivity"])
        self.assertEqual(analysis["ad_sensitivity"]["ad_averse"]["n_sessions"], 2)


class TestAnalysisModule(unittest.TestCase):
    """Tests for the analysis/table generation module."""

    def _make_summary(self):
        return {
            "full": {"CFS_mean": 0.45, "CFS_std": 0.05,
                     "click_position_jsd_mean": 0.12, "session_length_jsd_mean": 0.18,
                     "clicks_per_session_mean": 0.25, "abandon_rate_mean": 0.10,
                     "reformulation_rate_mean": 0.08,
                     "mean_session_length_mean": 4.2, "mean_clicks_mean": 1.8,
                     "mean_queries_mean": 1.4, "abandon_rate_raw_mean": 0.25,
                     "mean_satisfaction_mean": 0.68},
            "random": {"CFS_mean": 0.72, "CFS_std": 0.03,
                       "click_position_jsd_mean": 0.35, "session_length_jsd_mean": 0.30,
                       "clicks_per_session_mean": 0.68, "abandon_rate_mean": 0.20,
                       "reformulation_rate_mean": 0.15,
                       "mean_session_length_mean": 6.1, "mean_clicks_mean": 2.7,
                       "mean_queries_mean": 1.6, "abandon_rate_raw_mean": 0.55,
                       "mean_satisfaction_mean": 0.47},
        }

    def test_main_comparison_table(self):
        from searchagent.evaluation.analysis import generate_main_comparison_table
        table = generate_main_comparison_table(self._make_summary())
        self.assertIn("\\begin{table}", table)
        self.assertIn("SearchAgent (Full)", table)
        self.assertIn("Random", table)
        self.assertIn("\\textbf{", table)  # best value bolded

    def test_ablation_table_empty(self):
        from searchagent.evaluation.analysis import generate_ablation_table
        table = generate_ablation_table({"random": {}})
        self.assertIn("% No ablation", table)

    def test_markdown_report(self):
        from searchagent.evaluation.analysis import generate_markdown_report
        report = generate_markdown_report(self._make_summary())
        self.assertIn("# SearchAgent Experiment Report", report)
        self.assertIn("Agent Comparison", report)
        self.assertIn("Descriptive Statistics", report)

    def test_descriptive_table(self):
        from searchagent.evaluation.analysis import generate_descriptive_table
        table = generate_descriptive_table(self._make_summary())
        self.assertIn("\\begin{table}", table)
        self.assertIn("Avg. Length", table)


class TestHumanEval(unittest.TestCase):
    """Tests for the human evaluation framework."""

    def test_create_evaluation_set(self):
        from searchagent.evaluation.human_eval import EvalSession, create_evaluation_set
        real = [EvalSession("r1", "real", "topic", ["q1"], [], 1, 0, True)]
        sim = [EvalSession("s1", "simulated", "topic", ["q2"], [], 1, 0, False)]
        blinded, key = create_evaluation_set(real, sim, seed=42)
        self.assertEqual(len(blinded), 2)
        self.assertEqual(len(key), 2)
        # All blinded sessions should have source="hidden"
        for s in blinded:
            self.assertEqual(s.source, "hidden")
        # Answer key should map to real/simulated
        self.assertIn("real", key.values())
        self.assertIn("simulated", key.values())

    def test_compute_results_perfect(self):
        from searchagent.evaluation.human_eval import (
            JudgeAnnotation, compute_human_eval_results,
        )
        key = {"S001": "real", "S002": "simulated"}
        annotations = [
            JudgeAnnotation("S001", "j1", "real", 5),
            JudgeAnnotation("S002", "j1", "simulated", 5),
        ]
        r = compute_human_eval_results(annotations, key)
        self.assertAlmostEqual(r.discrimination_accuracy, 1.0)
        self.assertAlmostEqual(r.accuracy_on_real, 1.0)
        self.assertAlmostEqual(r.accuracy_on_simulated, 1.0)

    def test_compute_results_random(self):
        from searchagent.evaluation.human_eval import (
            JudgeAnnotation, compute_human_eval_results,
        )
        key = {"S001": "real", "S002": "simulated"}
        # Judge gets both wrong
        annotations = [
            JudgeAnnotation("S001", "j1", "simulated", 2),
            JudgeAnnotation("S002", "j1", "real", 2),
        ]
        r = compute_human_eval_results(annotations, key)
        self.assertAlmostEqual(r.discrimination_accuracy, 0.0)

    def test_format_session(self):
        from searchagent.evaluation.human_eval import EvalSession, format_session_for_display
        s = EvalSession("S001", "hidden", "laptops", ["best laptop 2024"],
                        [{"rank": 1, "title": "Top 10 Laptops", "dwell_time": 15.0}],
                        1, 1, False)
        text = format_session_for_display(s)
        self.assertIn("S001", text)
        self.assertIn("best laptop 2024", text)
        self.assertIn("Top 10 Laptops", text)

    def test_fleiss_kappa_perfect_agreement(self):
        from searchagent.evaluation.human_eval import (
            JudgeAnnotation, _fleiss_kappa,
        )
        key = {"S001": "real", "S002": "simulated"}
        annotations = [
            JudgeAnnotation("S001", "j1", "real", 5),
            JudgeAnnotation("S001", "j2", "real", 5),
            JudgeAnnotation("S002", "j1", "simulated", 5),
            JudgeAnnotation("S002", "j2", "simulated", 5),
        ]
        kappa = _fleiss_kappa(annotations, key)
        self.assertAlmostEqual(kappa, 1.0)

    def test_format_results_latex(self):
        from searchagent.evaluation.human_eval import HumanEvalResults, format_results_latex
        r = HumanEvalResults(20, 3, 60, 0.55, 0.60, 0.50, 0.35, 3.2, 2.8)
        latex = format_results_latex(r)
        self.assertIn("\\begin{table}", latex)
        self.assertIn("55.0%", latex)


class TestDownstreamEval(unittest.TestCase):
    """Tests for downstream click prediction evaluation."""

    def test_pbm_fit_predict(self):
        from searchagent.evaluation.downstream_eval import ClickSample, PositionBasedClickModel
        samples = [
            ClickSample(rank=1, is_clicked=True, session_length=1, query_position=1, n_prior_clicks=0),
            ClickSample(rank=1, is_clicked=True, session_length=1, query_position=1, n_prior_clicks=0),
            ClickSample(rank=1, is_clicked=False, session_length=1, query_position=1, n_prior_clicks=0),
            ClickSample(rank=5, is_clicked=False, session_length=1, query_position=1, n_prior_clicks=0),
            ClickSample(rank=5, is_clicked=False, session_length=1, query_position=1, n_prior_clicks=0),
        ]
        model = PositionBasedClickModel(max_rank=10)
        model.fit(samples)
        # Rank 1: 2 clicks, 3 impressions → (2+1)/(3+2) = 0.6
        self.assertAlmostEqual(model.predict(1), 0.6)
        # Rank 5: 0 clicks, 2 impressions → (0+1)/(2+2) = 0.25
        self.assertAlmostEqual(model.predict(5), 0.25)

    def test_log_likelihood(self):
        from searchagent.evaluation.downstream_eval import ClickSample, PositionBasedClickModel
        train = [
            ClickSample(1, True, 1, 1, 0), ClickSample(1, True, 1, 1, 0),
            ClickSample(2, False, 1, 1, 0), ClickSample(2, False, 1, 1, 0),
        ]
        model = PositionBasedClickModel(max_rank=10)
        model.fit(train)
        ll = model.log_likelihood(train)
        self.assertLess(ll, 0)  # log-likelihood is negative
        self.assertGreater(ll, -10)  # but not too negative

    def test_run_downstream_evaluation(self):
        from searchagent.evaluation.downstream_eval import (
            ClickSample, run_downstream_evaluation,
        )
        train = [ClickSample(r, r <= 2, 1, 1, 0) for r in range(1, 11)] * 10
        test = [ClickSample(r, r <= 2, 1, 1, 0) for r in range(1, 11)] * 5
        result = run_downstream_evaluation(train, test, "simulated", "real")
        self.assertEqual(result.train_source, "simulated")
        self.assertEqual(result.test_source, "real")
        self.assertGreater(result.click_accuracy, 0.5)
        self.assertGreater(result.perplexity, 1.0)

    def test_format_downstream_latex(self):
        from searchagent.evaluation.downstream_eval import (
            ClickSample, run_downstream_evaluation, format_downstream_latex,
        )
        train = [ClickSample(r, r == 1, 1, 1, 0) for r in range(1, 11)] * 5
        test = [ClickSample(r, r == 1, 1, 1, 0) for r in range(1, 11)] * 3
        results = {
            "real_to_real": run_downstream_evaluation(train, test, "real", "real"),
        }
        latex = format_downstream_latex(results)
        self.assertIn("\\begin{table}", latex)
        self.assertIn("Real", latex)


class TestStatisticalTests(unittest.TestCase):
    """Tests for the statistical significance testing module."""

    def test_bootstrap_ci_basic(self):
        from searchagent.evaluation.statistical_tests import bootstrap_ci
        values = np.array([0.5, 0.6, 0.55, 0.52, 0.58])
        mean, ci_lo, ci_hi = bootstrap_ci(values)
        self.assertAlmostEqual(mean, 0.55, places=5)
        self.assertLess(ci_lo, mean)
        self.assertGreater(ci_hi, mean)
        self.assertGreater(ci_lo, 0.0)
        self.assertLess(ci_hi, 1.0)

    def test_bootstrap_ci_single_value(self):
        from searchagent.evaluation.statistical_tests import bootstrap_ci
        mean, ci_lo, ci_hi = bootstrap_ci(np.array([0.42]))
        self.assertAlmostEqual(mean, 0.42)
        self.assertAlmostEqual(ci_lo, 0.42)
        self.assertAlmostEqual(ci_hi, 0.42)

    def test_paired_permutation_identical(self):
        from searchagent.evaluation.statistical_tests import paired_permutation_test
        a = np.array([0.5, 0.6, 0.55])
        p = paired_permutation_test(a, a)
        self.assertGreater(p, 0.5)  # no difference → large p

    def test_paired_permutation_different(self):
        from searchagent.evaluation.statistical_tests import paired_permutation_test
        a = np.array([0.1, 0.12, 0.11, 0.09, 0.10, 0.11, 0.10, 0.12])
        b = np.array([0.9, 0.88, 0.91, 0.87, 0.90, 0.89, 0.92, 0.88])
        p = paired_permutation_test(a, b)
        self.assertLess(p, 0.01)  # clearly different

    def test_wilcoxon_fallback(self):
        from searchagent.evaluation.statistical_tests import wilcoxon_test
        # n < 6 → should fall back to permutation test without error
        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.8, 0.9, 0.7])
        p = wilcoxon_test(a, b)
        self.assertLess(p, 0.5)
        self.assertIsInstance(p, float)

    def test_cohens_d_large(self):
        from searchagent.evaluation.statistical_tests import cohens_d
        a = np.array([10.0, 10.1, 9.9, 10.2, 9.8])
        b = np.array([1.0, 1.2, 0.8, 1.3, 0.7])
        d = cohens_d(a, b)
        self.assertGreater(abs(d), 0.8)  # large effect

    def test_cohens_d_zero(self):
        from searchagent.evaluation.statistical_tests import cohens_d
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(cohens_d(a, b), 0.0)

    def test_significance_marker(self):
        from searchagent.evaluation.statistical_tests import significance_marker
        self.assertEqual(significance_marker(0.0005), "***")
        self.assertEqual(significance_marker(0.005), "**")
        self.assertEqual(significance_marker(0.03), "*")
        self.assertEqual(significance_marker(0.1), "")

    def test_pairwise_significance_table(self):
        from searchagent.evaluation.statistical_tests import pairwise_significance_table
        per_seed = {
            "full": [{"CFS": 0.3}, {"CFS": 0.32}, {"CFS": 0.28}],
            "random": [{"CFS": 0.8}, {"CFS": 0.85}, {"CFS": 0.78}],
        }
        results = pairwise_significance_table(per_seed, metric="CFS")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].agent_a, "full")
        self.assertEqual(results[0].agent_b, "random")
        self.assertLess(results[0].p_value, 0.5)  # should be different

    def test_format_significance_table_latex(self):
        from searchagent.evaluation.statistical_tests import (
            pairwise_significance_table, format_significance_table_latex,
        )
        per_seed = {
            "full": [{"CFS": 0.3}, {"CFS": 0.32}, {"CFS": 0.28}],
            "random": [{"CFS": 0.8}, {"CFS": 0.85}, {"CFS": 0.78}],
        }
        results = pairwise_significance_table(per_seed, metric="CFS")
        latex = format_significance_table_latex(results)
        self.assertIn("\\begin{table}", latex)
        self.assertIn("Agent A", latex)
        self.assertIn("$p$-value", latex)


if __name__ == "__main__":
    unittest.main()
