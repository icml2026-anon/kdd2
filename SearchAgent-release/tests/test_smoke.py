"""Smoke tests for SearchAgent core components."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        from searchagent.config import SearchAgentConfig
        config = SearchAgentConfig()
        self.assertEqual(config.llm.model_name, "deepseek-v4-pro")
        self.assertEqual(len(config.intent.states), 3)
        self.assertEqual(config.environment.max_results_per_page, 10)

    def test_from_yaml(self):
        from searchagent.config import SearchAgentConfig
        yaml_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        if yaml_path.exists():
            config = SearchAgentConfig.from_yaml(str(yaml_path))
            self.assertIsNotNone(config.llm.model_name)
            self.assertGreater(len(config.persona.search_expertise), 0)


class TestPersona(unittest.TestCase):
    def test_persona_creation(self):
        from searchagent.agent.persona import Persona
        p = Persona(
            persona_id=0, age=30, search_expertise="intermediate",
            ad_sensitivity="ad_neutral", decision_style="deliberate",
            patience_level="medium", search_topic="best headphones",
            search_goal="find good headphones",
        )
        self.assertEqual(p.age, 30)
        desc = p.to_prompt_description()
        self.assertIn("30-year-old", desc)
        self.assertIn("headphones", desc)

    def test_persona_generator(self):
        from searchagent.agent.persona import PersonaGenerator
        from searchagent.config import PersonaConfig
        gen = PersonaGenerator(PersonaConfig(), seed=42)
        personas = gen.generate(5)
        self.assertEqual(len(personas), 5)
        ids = [p.persona_id for p in personas]
        self.assertEqual(ids, [0, 1, 2, 3, 4])
        for p in personas:
            self.assertIn(p.search_expertise, ["novice", "intermediate", "expert"])
            self.assertTrue(18 <= p.age <= 65)

    def test_persona_to_dict(self):
        from searchagent.agent.persona import Persona
        p = Persona(
            persona_id=1, age=25, search_expertise="expert",
            ad_sensitivity="ad_averse", decision_style="impulsive",
            patience_level="low",
        )
        d = p.to_dict()
        self.assertEqual(d["persona_id"], 1)
        self.assertEqual(d["ad_sensitivity"], "ad_averse")


class TestMemory(unittest.TestCase):
    def test_empty_memory(self):
        from searchagent.agent.memory import AgentMemory
        mem = AgentMemory()
        self.assertEqual(len(mem.entries), 0)
        ctx = mem.get_recent_context()
        self.assertIn("No previous", ctx)

    def test_add_entries(self):
        from searchagent.agent.memory import AgentMemory, MemoryEntry
        mem = AgentMemory()
        mem.add(MemoryEntry(step=0, query="test", action="click_organic", target="url1"))
        mem.add(MemoryEntry(step=1, query="test", action="skip"))
        mem.add(MemoryEntry(step=2, query="test", action="click_ad", target="ad1", is_ad=True))
        self.assertEqual(len(mem.entries), 3)
        self.assertEqual(len(mem.clicks), 2)
        self.assertEqual(len(mem.ad_clicks), 1)

    def test_session_summary(self):
        from searchagent.agent.memory import AgentMemory, MemoryEntry
        mem = AgentMemory()
        mem.add(MemoryEntry(step=0, query="q1", action="click_organic"))
        mem.add(MemoryEntry(step=1, query="q1", action="reformulate"))
        mem.add(MemoryEntry(step=2, query="q2", action="click_ad", is_ad=True))
        summary = mem.get_session_summary()
        self.assertEqual(summary["total_steps"], 3)
        self.assertEqual(summary["queries_issued"], 1)
        self.assertEqual(summary["ad_clicks"], 1)

    def test_reset(self):
        from searchagent.agent.memory import AgentMemory, MemoryEntry
        mem = AgentMemory()
        mem.add(MemoryEntry(step=0, query="q", action="skip"))
        mem.reset()
        self.assertEqual(len(mem.entries), 0)

    def test_max_entries(self):
        from searchagent.agent.memory import AgentMemory, MemoryEntry
        mem = AgentMemory(max_entries=3)
        for i in range(10):
            mem.add(MemoryEntry(step=i, query="q", action="skip"))
        self.assertEqual(len(mem.entries), 3)
        self.assertEqual(mem.entries[0].step, 7)


class TestSERP(unittest.TestCase):
    def test_search_result(self):
        from searchagent.environment.serp import SearchResult
        r = SearchResult(rank=1, title="Test", snippet="Snippet", url="http://test.com")
        display = r.to_display()
        self.assertIn("Test", display)
        self.assertNotIn("[AD]", display)

    def test_ad_result(self):
        from searchagent.environment.serp import SearchResult
        r = SearchResult(rank=1, title="Ad", snippet="Buy now", url="http://ad.com", is_ad=True)
        display = r.to_display()
        self.assertIn("Sponsored", display)

    def test_serp_page(self):
        from searchagent.environment.serp import SERPPage, SearchResult
        results = [
            SearchResult(rank=i+1, title=f"R{i}", snippet=f"S{i}", url=f"u{i}", is_ad=(i==0))
            for i in range(5)
        ]
        serp = SERPPage(query="test", page_number=1, results=results)
        self.assertEqual(len(serp.organic_results), 4)
        self.assertEqual(len(serp.ad_results), 1)
        display = serp.to_display()
        self.assertIn("test", display)

    def test_serp_generator(self):
        from searchagent.config import EnvironmentConfig
        from searchagent.environment.serp import SERPGenerator
        gen = SERPGenerator(EnvironmentConfig(), seed=42)
        serp = gen.generate("best noise cancelling headphones")
        self.assertGreater(len(serp.results), 0)
        self.assertTrue(any(r.is_ad for r in serp.results))
        self.assertTrue(any(not r.is_ad for r in serp.results))

    def test_trec_click_rank_and_timestamp(self):
        from searchagent.environment.serp import TRECSERPLoader
        xml = """<sessiontrack2012><session num="1"><topic num="1"><desc>x</desc></topic><interaction num="1"><query>q</query><results><result rank="1"><title>a</title></result><result rank="2"><title>b</title></result></results><clicked><click num="1" starttime="09:54:42.635247" endtime="09:54:43.722679"><rank>2</rank></click></clicked></interaction></session></sessiontrack2012>"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trec.xml"
            path.write_text(xml)
            session = TRECSERPLoader(str(path)).sessions[0]
        self.assertEqual(session.click_positions, [2])
        self.assertAlmostEqual(session.interactions[0].clicks[0].dwell_time, 1.087432, places=6)

    def test_replay_paging(self):
        from searchagent.config import EnvironmentConfig
        from searchagent.environment.search_env import ReplayEnvironment
        from searchagent.environment.serp import SearchResult, TRECInteraction, TRECSession
        results = [SearchResult(rank=i, title=str(i), snippet="", url="") for i in range(1, 26)]
        session = TRECSession(1, 0, 0, "", [TRECInteraction(1, "q", "", 0.0, results, [])], "")
        env = ReplayEnvironment(session, EnvironmentConfig(max_results_per_page=10, max_serp_pages=3))
        page = env.reset(1, "q")
        self.assertEqual([r.rank for r in page.results], list(range(1, 11)))
        page, _ = env.step("next_page")
        self.assertEqual([r.rank for r in page.results], list(range(11, 21)))
        page, _ = env.step("next_page")
        self.assertEqual([r.rank for r in page.results], list(range(21, 26)))

    def test_serp_generator_default_query(self):
        from searchagent.config import EnvironmentConfig
        from searchagent.environment.serp import SERPGenerator
        gen = SERPGenerator(EnvironmentConfig(), seed=42)
        serp = gen.generate("random unknown query xyz")
        self.assertGreater(len(serp.results), 0)


class TestSearchEnvironment(unittest.TestCase):
    def test_reset(self):
        from searchagent.config import EnvironmentConfig
        from searchagent.environment.search_env import SearchEnvironment
        env = SearchEnvironment(EnvironmentConfig())
        serp = env.reset(persona_id=0, initial_query="test query")
        self.assertIsNotNone(serp)
        self.assertEqual(serp.query, "test query")

    def test_click_action(self):
        from searchagent.config import EnvironmentConfig
        from searchagent.environment.search_env import SearchEnvironment
        env = SearchEnvironment(EnvironmentConfig())
        env.reset(0, "test")
        serp, done = env.step("click_organic", click_index=0)
        self.assertFalse(done)
        record = env.get_session_record()
        self.assertEqual(record.total_steps, 1)
        self.assertEqual(len(record.clicks), 1)

    def test_abandon_action(self):
        from searchagent.config import EnvironmentConfig
        from searchagent.environment.search_env import SearchEnvironment
        env = SearchEnvironment(EnvironmentConfig())
        env.reset(0, "test")
        serp, done = env.step("abandon")
        self.assertTrue(done)
        self.assertIsNone(serp)
        record = env.get_session_record()
        self.assertTrue(record.abandoned)

    def test_next_page(self):
        from searchagent.config import EnvironmentConfig
        from searchagent.environment.search_env import SearchEnvironment
        env = SearchEnvironment(EnvironmentConfig())
        env.reset(0, "test")
        serp, done = env.step("next_page")
        self.assertFalse(done)
        self.assertEqual(serp.page_number, 2)

    def test_reformulate(self):
        from searchagent.config import EnvironmentConfig
        from searchagent.environment.search_env import SearchEnvironment
        env = SearchEnvironment(EnvironmentConfig())
        env.reset(0, "test")
        serp, done = env.step("reformulate", new_query="better test")
        self.assertFalse(done)
        self.assertEqual(serp.query, "better test")
        record = env.get_session_record()
        self.assertIn("better test", record.queries)

    def test_session_record_properties(self):
        from searchagent.config import EnvironmentConfig
        from searchagent.environment.search_env import SearchEnvironment
        env = SearchEnvironment(EnvironmentConfig())
        env.reset(0, "test")
        env.step("click_organic", click_index=0)
        env.step("click_ad", click_index=1)
        env.step("reformulate", new_query="new q")
        record = env.get_session_record()
        self.assertEqual(record.total_steps, 3)
        self.assertEqual(record.num_queries, 2)
        d = record.to_dict()
        self.assertIn("total_steps", d)


class TestIntentTracker(unittest.TestCase):
    def test_initial_distribution(self):
        from searchagent.agent.search_agent import IntentTracker
        from searchagent.config import IntentConfig
        tracker = IntentTracker(IntentConfig())
        self.assertAlmostEqual(tracker.distribution.sum(), 1.0)

    def test_update(self):
        from searchagent.agent.search_agent import IntentTracker
        from searchagent.config import IntentConfig
        config = IntentConfig(
            transitions={"click_ad": [0.1, 0.2, 0.7]}
        )
        tracker = IntentTracker(config)
        old_dist = tracker.distribution.copy()
        tracker.update("click_ad")
        # Transactional intent should increase after ad click
        self.assertGreater(tracker.distribution[2], old_dist[2])
        self.assertAlmostEqual(tracker.distribution.sum(), 1.0)

    def test_describe(self):
        from searchagent.agent.search_agent import IntentTracker
        from searchagent.config import IntentConfig
        tracker = IntentTracker(IntentConfig())
        desc = tracker.describe()
        self.assertIn("Intent:", desc)
        self.assertIn("exploratory", desc)

    def test_dominant_intent(self):
        from searchagent.agent.search_agent import IntentTracker
        from searchagent.config import IntentConfig
        tracker = IntentTracker(IntentConfig(
            initial_distribution=[0.1, 0.1, 0.8]
        ))
        self.assertEqual(tracker.dominant_intent(), "transactional")


class TestPrompts(unittest.TestCase):
    def test_action_selection_prompt(self):
        from searchagent.agent.prompts import build_action_selection_prompt
        prompt = build_action_selection_prompt(
            persona_desc="You are a 30-year-old user.",
            current_query="test query",
            serp_description="1. Result one\n2. Result two",
            memory_context="No previous actions.",
            intent_description="Intent: exploratory (60%)",
        )
        self.assertIn("test query", prompt)
        self.assertIn("ANSWER_JSON:", prompt)
        self.assertIn("A = click_result", prompt)
        self.assertIn("G = stop_unsatisfied", prompt)

    def test_query_generation_prompt(self):
        from searchagent.agent.prompts import build_query_generation_prompt
        prompt = build_query_generation_prompt(
            persona_desc="You are a user.",
            original_query="old query",
            memory_context="No previous.",
            intent_description="focused",
            is_reformulation=True,
        )
        self.assertIn("old query", prompt)
        self.assertIn("REFORMULATE", prompt)

    def test_macro_actions(self):
        from searchagent.agent.prompts import MACRO_ACTIONS, ACTION_LABELS
        self.assertEqual(len(MACRO_ACTIONS), 7)
        self.assertEqual(len(ACTION_LABELS), 7)


class TestMetrics(unittest.TestCase):
    def _make_sessions(self, n=20):
        from searchagent.environment.search_env import ClickRecord, SessionRecord
        sessions = []
        rng = np.random.RandomState(42)
        for i in range(n):
            n_clicks = rng.randint(0, 5)
            clicks = []
            for _ in range(n_clicks):
                rank = rng.choice(range(1, 11), p=[0.3, 0.2, 0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.06, 0.05])
                clicks.append(ClickRecord(
                    step=0, query="q", rank=rank, title="t",
                    is_ad=(rng.random() < 0.1), relevance=0.5,
                ))
            n_queries = rng.randint(1, 4)
            sessions.append(SessionRecord(
                persona_id=i, initial_query="q",
                clicks=clicks,
                queries=[f"q{j}" for j in range(n_queries)],
                total_steps=rng.randint(3, 15),
                abandoned=(rng.random() < 0.15),
            ))
        return sessions

    def test_click_position_jsd(self):
        from searchagent.evaluation.metrics import click_position_jsd
        sessions = self._make_sessions()
        jsd = click_position_jsd(sessions)
        self.assertGreaterEqual(jsd, 0.0)
        self.assertTrue(np.isfinite(jsd))

    def test_session_length_jsd(self):
        from searchagent.evaluation.metrics import session_length_jsd
        sessions = self._make_sessions()
        jsd = session_length_jsd(sessions)
        self.assertGreaterEqual(jsd, 0.0)

    def test_clicks_per_session_gap(self):
        from searchagent.evaluation.metrics import clicks_per_session_gap
        sessions = self._make_sessions()
        gap = clicks_per_session_gap(sessions)
        self.assertGreaterEqual(gap, 0.0)

    def test_reformulation_rate(self):
        from searchagent.evaluation.metrics import reformulation_rate_gap
        sessions = self._make_sessions()
        gap = reformulation_rate_gap(sessions)
        self.assertGreaterEqual(gap, 0.0)

    def test_abandon_rate(self):
        from searchagent.evaluation.metrics import abandon_rate_gap
        sessions = self._make_sessions()
        gap = abandon_rate_gap(sessions)
        self.assertGreaterEqual(gap, 0.0)

    def test_composite_fidelity_score(self):
        from searchagent.evaluation.metrics import composite_fidelity_score
        sessions = self._make_sessions()
        result = composite_fidelity_score(sessions)
        self.assertIn("CFS", result)
        self.assertIn("click_position_jsd", result)
        self.assertGreaterEqual(result["CFS"], 0.0)
        self.assertEqual(result["num_sessions"], 20)


    def test_saved_session_roundtrip_preserves_feedback(self):
        from scripts.evaluate import load_sessions
        from searchagent.environment.search_env import ClickRecord, SessionRecord
        session = SessionRecord(
            persona_id=1,
            initial_query="q",
            clicks=[ClickRecord(step=0, query="q", rank=2, title="t", is_ad=False, relevance=0.4, satisfaction=4.25, dwell_time=73.5)],
            queries=["q"],
            total_steps=1,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sessions.json"
            path.write_text(json.dumps([session.to_dict()]))
            loaded = load_sessions(str(path))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(loaded[0].clicks), 1)
        self.assertAlmostEqual(loaded[0].clicks[0].satisfaction, 4.25)
        self.assertAlmostEqual(loaded[0].clicks[0].dwell_time, 73.5)
    def test_empty_sessions(self):
        from searchagent.evaluation.metrics import composite_fidelity_score
        result = composite_fidelity_score([])
        self.assertIn("CFS", result)


class TestLLMBackbone(unittest.TestCase):
    def test_init(self):
        from searchagent.agent.llm_backbone import LLMBackbone
        from searchagent.config import LLMConfig
        llm = LLMBackbone(LLMConfig(model_name="dummy"))
        self.assertFalse(llm._loaded)


    def test_json_output_payload(self):
        from searchagent.agent.llm_backbone import LLMBackbone
        from searchagent.config import LLMConfig
        llm = LLMBackbone(LLMConfig(model_name="dummy"))
        payloads = []
        def fake_post(payload):
            payloads.append(payload)
            labels = list("ABCDEFG")
            return {"choices": [{"message": {"content": json.dumps({"probabilities": {label: 1 / len(labels) for label in labels}})}}]}
        llm._post = fake_post
        probs = llm.get_action_logprobs("Return JSON for A-G", list("ABCDEFG"))
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertEqual(payloads[0]["response_format"], {"type": "json_object"})

    def test_query_json_output_payload(self):
        from searchagent.agent.llm_backbone import LLMBackbone
        from searchagent.config import LLMConfig
        llm = LLMBackbone(LLMConfig(model_name="dummy"))
        payloads = []
        def fake_post(payload):
            payloads.append(payload)
            return {"choices": [{"message": {"content": json.dumps({"query": "next query"})}}]}
        llm._post = fake_post
        query = llm.generate_query("Generate JSON", "old query")
        self.assertEqual(query, "next query")
        self.assertEqual(payloads[0]["response_format"], {"type": "json_object"})
    def test_build_chat_prompt_deferred(self):
        """Chat prompt building requires loaded model; just test init."""
        from searchagent.agent.llm_backbone import LLMBackbone
        from searchagent.config import LLMConfig
        llm = LLMBackbone(LLMConfig(model_name="dummy"))
        self.assertIsNone(llm.tokenizer)


class TestSearchUserAgent(unittest.TestCase):
    def test_init(self):
        from searchagent.agent.persona import Persona
        from searchagent.agent.search_agent import SearchUserAgent
        from searchagent.agent.llm_backbone import LLMBackbone
        from searchagent.config import SearchAgentConfig, LLMConfig

        config = SearchAgentConfig(llm=LLMConfig(model_name="dummy"))
        persona = Persona(
            persona_id=0, age=30, search_expertise="intermediate",
            ad_sensitivity="ad_neutral", decision_style="deliberate",
            patience_level="medium", search_topic="test",
        )
        llm = LLMBackbone(config.llm)
        agent = SearchUserAgent(persona=persona, llm=llm, config=config)
        self.assertEqual(agent.current_query, "test")
        self.assertEqual(agent.step_count, 0)


class TestPaperBaselineAdapters(unittest.TestCase):
    def test_exact_adapter_rules(self):
        from searchagent.agent.baselines import BaselineAdapter
        self.assertEqual(BaselineAdapter.cot("A"), "click_result")
        self.assertEqual(BaselineAdapter.cot("G"), "stop_unsatisfied")
        self.assertEqual(BaselineAdapter.usersimcrs("inspect", has_item=True), "click_result")
        self.assertEqual(BaselineAdapter.usersimcrs("reject"), "skip")
        self.assertEqual(BaselineAdapter.usersimcrs("request_more"), "next_page")
        self.assertEqual(BaselineAdapter.usersimcrs("clarify"), "reformulate")
        self.assertEqual(BaselineAdapter.usersimcrs("end", satisfaction=0.5), "stop_satisfied")
        self.assertEqual(BaselineAdapter.usersimcrs("end", satisfaction=0.49), "stop_unsatisfied")
        self.assertEqual(BaselineAdapter.cshi("continue"), "skip")
        self.assertEqual(BaselineAdapter.cshi("refine"), "reformulate")
        self.assertEqual(BaselineAdapter.cshi("switch"), "new_query")
        self.assertEqual(BaselineAdapter.cshi("finish", 0.5), "stop_satisfied")
        self.assertEqual(BaselineAdapter.cshi("finish", 0.49), "stop_unsatisfied")
        self.assertEqual(BaselineAdapter.agent4rec("browsing", page_boundary=False), "skip")
        self.assertEqual(BaselineAdapter.agent4rec("browsing", page_boundary=True), "next_page")
        self.assertEqual(BaselineAdapter.agent4rec("exit", exit_rating="satisfied"), "stop_satisfied")
        self.assertEqual(BaselineAdapter.agent4rec("exit", exit_rating="unsatisfied"), "stop_unsatisfied")
        self.assertEqual(BaselineAdapter.simuser("request_more"), "next_page")
        self.assertEqual(BaselineAdapter.simuser("switch_goal"), "new_query")

    def test_query_change_rules(self):
        from searchagent.agent.baselines import BaselineAdapter, normalized_levenshtein_distance, retained_token_fraction
        self.assertLess(normalized_levenshtein_distance("cheap flights tokyo", "cheap flight tokyo"), 0.5)
        self.assertGreaterEqual(normalized_levenshtein_distance("cheap flights tokyo", "python tutorial"), 0.5)
        self.assertEqual(BaselineAdapter.usimagent("query", "cheap flights tokyo", "cheap flight tokyo"), "reformulate")
        self.assertEqual(BaselineAdapter.usimagent("query", "cheap flights tokyo", "python tutorial"), "new_query")
        self.assertEqual(BaselineAdapter.bases("stop", terminal_utility=3), "stop_satisfied")
        self.assertEqual(BaselineAdapter.bases("stop", terminal_utility=2.99), "stop_unsatisfied")
        self.assertEqual(retained_token_fraction("a b c d", "a b x"), 0.5)
        self.assertEqual(BaselineAdapter.convsearch("a b c d", "a b x"), "reformulate")
        self.assertEqual(BaselineAdapter.convsearch("a b c d", "a x"), "new_query")

    def test_invalid_command_retry_then_fallback(self):
        from searchagent.agent.baselines import ReActSearchAgent
        from searchagent.agent.persona import Persona
        from searchagent.config import SearchAgentConfig

        class FakeLLM:
            def __init__(self, responses):
                self.responses = list(responses)
                self.calls = 0
            def generate_text(self, prompt, max_new_tokens=None):
                self.calls += 1
                return self.responses.pop(0) if self.responses else ""

        persona = Persona(persona_id=0, age=30, search_expertise="intermediate", ad_sensitivity="ad_neutral", decision_style="deliberate", patience_level="medium", search_topic="test")
        llm = FakeLLM(["invalid", "invalid"])
        agent = ReActSearchAgent(persona, llm, SearchAgentConfig())
        action, _ = agent.decide_action({"query": "test", "page": 1, "results": []})
        self.assertEqual(action, "stop_unsatisfied")
        self.assertEqual(llm.calls, 2)

        llm = FakeLLM(["invalid", "ACTION: NEXT"])
        agent = ReActSearchAgent(persona, llm, SearchAgentConfig())
        action, _ = agent.decide_action({"query": "test", "page": 1, "results": []})
        self.assertEqual(action, "next_page")
        self.assertEqual(llm.calls, 2)

    def test_react_click_rank(self):
        from searchagent.agent.baselines import ReActSearchAgent
        from searchagent.agent.persona import Persona
        from searchagent.config import SearchAgentConfig
        from searchagent.environment.serp import SearchResult

        class FakeLLM:
            def generate_text(self, prompt, max_new_tokens=None):
                return "Thought: internal\nACTION: CLICK(2)"

        persona = Persona(persona_id=0, age=30, search_expertise="intermediate", ad_sensitivity="ad_neutral", decision_style="deliberate", patience_level="medium", search_topic="test")
        agent = ReActSearchAgent(persona, FakeLLM(), SearchAgentConfig())
        action, _ = agent.decide_action({"query": "test", "page": 1, "results": []})
        results = [SearchResult(rank=1, title="a", snippet="", url=""), SearchResult(rank=2, title="b", snippet="", url="")]
        self.assertEqual(action, "click_result")
        self.assertEqual(agent.select_click_target({}, action, results), 1)

    def test_bases_sequence_is_consumed_sequentially(self):
        from searchagent.agent.baselines import BASESAgent
        from searchagent.agent.persona import Persona
        from searchagent.config import SearchAgentConfig

        class FakeLLM:
            def __init__(self):
                self.calls = 0
            def generate_text(self, prompt, max_new_tokens=None):
                self.calls += 1
                return json.dumps({"events": [{"event": "browse", "mode": "skip"}, {"event": "stop", "terminal_utility": 4}]})

        persona = Persona(persona_id=0, age=30, search_expertise="intermediate", ad_sensitivity="ad_neutral", decision_style="deliberate", patience_level="medium", search_topic="test")
        llm = FakeLLM()
        agent = BASESAgent(persona, llm, SearchAgentConfig())
        first, _ = agent.decide_action({"query": "test", "page": 1, "results": []})
        agent.record_action(first)
        second, _ = agent.decide_action({"query": "test", "page": 1, "results": []})
        self.assertEqual(first, "skip")
        self.assertEqual(second, "stop_satisfied")
        self.assertEqual(llm.calls, 1)

    def test_paper_baseline_registry(self):
        from scripts.run_experiment import PAPER_BASELINES
        self.assertEqual(PAPER_BASELINES, ["react", "polyskill", "reflexion", "cot", "tot", "usersimcrs_v2", "cshi", "usimagent", "bases", "agent4rec", "simuser", "convsearch_r1"])


if __name__ == "__main__":
    unittest.main()
