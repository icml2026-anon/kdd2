"""Session simulation loop orchestrating agent-environment interaction."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tqdm import tqdm

from searchagent.agent.llm_backbone import LLMBackbone
from searchagent.agent.persona import Persona, PersonaGenerator
from searchagent.agent.search_agent import SearchUserAgent
from searchagent.agent.prompts import MACRO_ACTIONS
from searchagent.config import SearchAgentConfig
from searchagent.environment.search_env import SearchEnvironment, SessionRecord

logger = logging.getLogger(__name__)


class SessionSimulator:
    """Runs search session simulations with LLM-powered user agents."""

    def __init__(self, config: SearchAgentConfig):
        self.config = config
        self.llm = LLMBackbone(config.llm)
        self.results: List[SessionRecord] = []

    def simulate_session(
        self,
        persona: Persona,
        env: SearchEnvironment,
    ) -> SessionRecord:
        """Simulate a single search session for a given persona."""
        agent = SearchUserAgent(persona=persona, llm=self.llm, config=self.config)

        # Initialize environment
        serp = env.reset(persona.persona_id, persona.search_topic)
        done = False
        max_steps = self.config.simulation.max_session_steps

        timeout_prob = self.config.simulation.session_timeout_prob

        while not done and agent.step_count < max_steps:
            serp_text = serp.to_display() if serp else "No results available."

            # Random session timeout (models real-world abandon behavior)
            # TREC 2014: ~49% abandon rate; this adds stochastic abandonment
            if agent.step_count > 0 and np.random.random() < timeout_prob:
                timeout_action = "stop_unsatisfied" if "stop_unsatisfied" in MACRO_ACTIONS else "abandon"
                agent.record_action(action=timeout_action)
                serp, done = env.step(action=timeout_action)
                break

            # Agent decides action
            action, probs = agent.decide_action(serp_text)

            # Handle click actions: select specific target
            click_index = None
            new_query = None
            pending_satisfaction = None
            pending_dwell = None

            if action in ("click_organic", "click_ad", "click_result"):
                click_index = agent.select_click_target(
                    serp_text, action,
                    serp_results=serp.results if serp else [],
                )
                if serp and 0 <= click_index < len(serp.results):
                    result = serp.results[click_index]
                    satisfaction = 3.0
                    dwell_time = 25.0
                    if hasattr(agent, "rate_satisfaction"):
                        try:
                            satisfaction = agent.rate_satisfaction(
                                result.title, result.snippet, result.is_ad,
                            )
                            dwell_time = agent.estimate_dwell_time(
                                satisfaction, result.is_ad,
                            )
                        except Exception:
                            pass
                    agent.record_action(
                        action=action,
                        target=result.url,
                        result_title=result.title,
                        is_ad=result.is_ad,
                        satisfaction=satisfaction,
                    )
                    pending_satisfaction = satisfaction
                    pending_dwell = dwell_time
                else:
                    agent.record_action(action=action)

            elif action in ("reformulate", "new_query"):
                is_reform = action == "reformulate"
                new_query = agent.generate_query(is_reformulation=is_reform)
                agent.current_query = new_query
                agent.record_action(action=action, target=new_query)

            else:
                agent.record_action(action=action)

            # Execute action in environment
            serp, done = env.step(
                action=action,
                click_index=click_index,
                new_query=new_query,
            )

            # Attach satisfaction/dwell AFTER env.step records the click
            if pending_satisfaction is not None:
                env.update_last_click(pending_satisfaction, pending_dwell)

        return env.get_session_record()

    def run(
        self,
        num_sessions: Optional[int] = None,
        personas: Optional[List[Persona]] = None,
    ) -> List[SessionRecord]:
        """Run multiple simulation sessions.

        Args:
            num_sessions: Number of sessions to simulate.
            personas: Pre-generated personas. If None, generates automatically.

        Returns:
            List of SessionRecord objects.
        """
        if num_sessions is None:
            num_sessions = self.config.simulation.num_sessions

        # Generate personas if not provided
        if personas is None:
            gen = PersonaGenerator(
                self.config.persona, seed=self.config.simulation.seed
            )
            personas = gen.generate(num_sessions)

        # Load LLM
        self.llm.load()

        self.results = []
        env = SearchEnvironment(
            self.config.environment, seed=self.config.simulation.seed
        )

        logger.info("Starting simulation: %d sessions", num_sessions)
        t0 = time.time()

        for i, persona in enumerate(tqdm(personas[:num_sessions], desc="Simulating")):
            try:
                record = self.simulate_session(persona, env)
                self.results.append(record)
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "Session %d/%d done (%.1fs elapsed, %.2fs/session)",
                        i + 1, num_sessions, elapsed, elapsed / (i + 1),
                    )
            except Exception as e:
                logger.error("Session %d failed: %s", i, e)

        elapsed = time.time() - t0
        logger.info(
            "Simulation complete: %d sessions in %.1fs (%.2fs/session)",
            len(self.results), elapsed,
            elapsed / max(len(self.results), 1),
        )
        return self.results

    def save_results(self, output_path: str) -> None:
        """Save simulation results to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s (%d sessions)", path, len(data))
