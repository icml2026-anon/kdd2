from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from searchagent.agent.llm_backbone import LLMBackbone
from searchagent.agent.persona import Persona, PersonaGenerator
from searchagent.agent.search_agent import SearchUserAgent
from searchagent.config import SearchAgentConfig
from searchagent.environment.search_env import SearchEnvironment, SessionRecord


logger = logging.getLogger(__name__)


class SessionSimulator:
    def __init__(self, config: SearchAgentConfig, calibrator=None):
        self.config = config
        self.llm = LLMBackbone(config.llm)
        self.calibrator = calibrator if calibrator is not None else self._load_calibrator()
        self.results: List[SessionRecord] = []

    def _load_calibrator(self):
        path = self.config.calibration.calibrator_path
        if not path or not Path(path).exists():
            return None
        import torch
        from searchagent.agent.calibration import Calibrator
        ckpt = torch.load(path, map_location="cpu")
        dims = ckpt.get("dims", {})
        model = Calibrator(
            n_intent=dims.get("n_intent", len(self.config.intent.states)),
            hidden1=dims.get("hidden1", dims.get("hidden", self.config.calibration.hidden1)),
            hidden2=dims.get("hidden2", self.config.calibration.hidden2),
            dropout=dims.get("dropout", self.config.calibration.dropout),
            alpha_max=dims.get("alpha_max", self.config.calibration.alpha_max),
            exam_dim=dims.get("exam_dim", self.config.calibration.exam_dim),
            attr_dim=dims.get("attr_dim", self.config.calibration.attr_dim),
            epsilon=dims.get("epsilon", self.config.calibration.epsilon),
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        return model

    def simulate_session(self, persona: Persona, env: SearchEnvironment) -> SessionRecord:
        agent = SearchUserAgent(persona=persona, llm=self.llm, config=self.config, calibrator=self.calibrator)
        serp = env.reset(persona.persona_id, persona.search_topic)
        done = False
        while not done and agent.step_count < self.config.simulation.max_session_steps:
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
                    satisfaction = agent.rate_satisfaction(result.title, result.snippet, result.is_ad, result.url)
                    dwell_time = agent.estimate_dwell_time(satisfaction, result.is_ad)
                    agent.record_action(
                        action=action,
                        target=result.url,
                        result_title=result.title,
                        is_ad=result.is_ad,
                        satisfaction=satisfaction,
                        dwell_time=dwell_time,
                    )
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

    def run(self, num_sessions: Optional[int] = None, personas: Optional[List[Persona]] = None) -> List[SessionRecord]:
        if num_sessions is None:
            num_sessions = self.config.simulation.num_sessions
        if personas is None:
            personas = PersonaGenerator(self.config.persona, seed=self.config.simulation.seed).generate(num_sessions)
        self.llm.load()
        self.results = []
        env = SearchEnvironment(self.config.environment, seed=self.config.simulation.seed)
        t0 = time.time()
        for persona in tqdm(personas[:num_sessions], desc="Simulating"):
            try:
                self.results.append(self.simulate_session(persona, env))
            except Exception as exc:
                logger.error("Session %s failed: %s", persona.persona_id, exc)
        logger.info("Simulation complete: %d sessions in %.1fs", len(self.results), time.time() - t0)
        return self.results

    def save_results(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([record.to_dict() for record in self.results], f, indent=2, ensure_ascii=False)
