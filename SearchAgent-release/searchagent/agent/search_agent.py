from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from searchagent.agent.llm_backbone import LLMBackbone
from searchagent.agent.memory import AgentMemory, MemoryEntry
from searchagent.agent.persona import Persona
from searchagent.agent.prompts import (
    ACTION_LABELS,
    MACRO_ACTIONS,
    QUALITY_LABELS,
    UTILITY_LABELS,
    build_action_selection_prompt,
    build_click_selection_prompt,
    build_query_generation_prompt,
    build_satisfaction_prompt,
    build_serp_quality_prompt,
)
from searchagent.config import IntentConfig, SearchAgentConfig


logger = logging.getLogger(__name__)


class CognitiveSearchState:
    def __init__(self):
        self.reset()

    def update(self, action: str, satisfaction: float = 0.0) -> None:
        if action in ("click_organic", "click_ad", "click_result"):
            u = float(np.clip((satisfaction - 1.0) / 4.0, 0.0, 1.0))
            delta = np.array([0.08 * u, 0.12 * u, 0.05 * (1.0 - u) - 0.08 * u, 0.03], dtype=np.float64)
        elif action == "skip":
            delta = np.array([-0.01, 0.0, 0.03, -0.02], dtype=np.float64)
        elif action == "next_page":
            delta = np.array([-0.02, 0.0, 0.05, -0.04], dtype=np.float64)
        elif action == "reformulate":
            delta = np.array([0.06, 0.0, -0.02, 0.04], dtype=np.float64)
        elif action == "new_query":
            delta = np.array([-0.03, 0.0, -0.01, -0.06], dtype=np.float64)
        else:
            delta = np.zeros(4, dtype=np.float64)
        state = np.array([self.need_clarity, self.knowledge_state, self.frustration, self.exploration_focus], dtype=np.float64)
        state = np.clip(state + delta, 0.0, 1.0)
        self.need_clarity, self.knowledge_state, self.frustration, self.exploration_focus = map(float, state)
        self._interaction_count += 1

    def get_action_modifiers(self) -> Dict[str, float]:
        return {a: 1.0 for a in MACRO_ACTIONS}

    def to_dict(self) -> Dict[str, float]:
        return {
            "need_clarity": self.need_clarity,
            "knowledge": self.knowledge_state,
            "frustration": self.frustration,
            "focus": self.exploration_focus,
        }

    def describe(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def reset(self) -> None:
        self.need_clarity = 0.3
        self.knowledge_state = 0.0
        self.frustration = 0.0
        self.exploration_focus = 0.3
        self._interaction_count = 0


class IntentTracker:
    def __init__(self, config: IntentConfig):
        self.states = list(config.states)
        self.transitions = dict(config.transitions)
        self.momentum = float(config.momentum)
        self.initial_distribution = np.asarray(config.initial_distribution, dtype=np.float64)
        self.reset()

    def update(self, action: str, momentum: Optional[float] = None) -> None:
        target_values = self.transitions.get(action)
        if target_values is None and action in ("click_organic", "click_ad"):
            target_values = self.transitions.get("click_result")
        if not target_values:
            return
        target = np.asarray(target_values, dtype=np.float64)
        if target.shape != self.distribution.shape or target.sum() <= 0:
            return
        target = target / target.sum()
        lam = self.momentum if momentum is None else float(momentum)
        self.distribution = lam * self.distribution + (1.0 - lam) * target
        self.distribution /= self.distribution.sum()

    def describe(self) -> str:
        return "Intent: " + ", ".join(f"{name} ({prob:.0%})" for name, prob in zip(self.states, self.distribution))

    def dominant_intent(self) -> str:
        return self.states[int(np.argmax(self.distribution))]

    def reset(self, initial: Optional[List[float]] = None) -> None:
        base = np.asarray(initial if initial is not None else self.initial_distribution, dtype=np.float64)
        self.distribution = base / base.sum()


class SearchUserAgent:
    def __init__(self, persona: Persona, llm: LLMBackbone, config: SearchAgentConfig, calibrator=None, dataset_stats=None):
        self.persona = persona
        self.llm = llm
        self.config = config
        self.memory = AgentMemory(max_entries=8)
        self.intent = IntentTracker(config.intent)
        self.cognitive_state = CognitiveSearchState()
        self.current_query = persona.search_topic
        self.step_count = 0
        self._persona_desc = persona.to_dict()
        self._last_satisfaction = 0.0
        self._total_clicks = 0
        self._total_skips = 0
        self.calibrator = calibrator
        from searchagent.agent.calibration import StateFeaturizer
        self._featurizer = StateFeaturizer(n_intent=len(config.intent.states))

    def _step_norm(self) -> float:
        denom = max(self.config.simulation.max_session_steps - 1, 1)
        return float(np.clip(self.step_count / denom, 0.0, 1.0))

    def _page_norm(self) -> float:
        page = min(1 + self.memory.pages_viewed, self.config.environment.max_serp_pages)
        denom = max(self.config.environment.max_serp_pages - 1, 1)
        return float(np.clip((page - 1) / denom, 0.0, 1.0))

    def _click_norm(self) -> float:
        return float(np.clip(self._total_clicks / max(self.config.simulation.max_session_steps, 1), 0.0, 1.0))

    def _skip_norm(self) -> float:
        return float(np.clip(self._total_skips / max(self.config.simulation.max_session_steps, 1), 0.0, 1.0))

    def _serp_quality(self, serp_description: str) -> float:
        prompt = build_serp_quality_prompt(self.current_query, serp_description)
        probs = self.llm.get_action_logprobs(prompt, QUALITY_LABELS, rotation_key=f"quality:{self.persona.persona_id}:{self.step_count}")
        return float(sum((i / 10.0) * probs.get(f"Q{i}", 0.0) for i in range(11)))

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        prompt = build_action_selection_prompt(
            persona_desc=self._persona_desc,
            current_query=self.current_query,
            serp_description=serp_description,
            memory_context=self.memory.get_recent_records(),
            intent_description=self.intent.distribution.tolist(),
            css_context=self.cognitive_state.to_dict(),
            session_meta={"step": self.step_count, "page": 1 + self.memory.pages_viewed},
        )
        action_probs = self.llm.get_action_logprobs(prompt, ACTION_LABELS, rotation_key=f"action:{self.persona.persona_id}:{self.step_count}")
        return self._decide_action_calibrated(serp_description, action_probs)

    def _decide_action_calibrated(self, serp_description: str, action_probs: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        from searchagent.agent.calibration import CALIB_ACTIONS
        prior_macro = {action: action_probs.get(label, 0.0) for action, label in zip(MACRO_ACTIONS, ACTION_LABELS)}
        prior = np.asarray([prior_macro.get(action, 0.0) for action in CALIB_ACTIONS], dtype=np.float64)
        prior = np.clip(prior, self.config.calibration.epsilon, None)
        prior /= prior.sum()
        if self.calibrator is None:
            probs = prior
        else:
            quality = self._serp_quality(serp_description)
            memory_feats = {
                "n_click": self._click_norm(),
                "n_skip": self._skip_norm(),
                "last_satisfaction": float(np.clip((self._last_satisfaction - 1.0) / 4.0, 0.0, 1.0)) if self._last_satisfaction > 0 else 0.0,
                "page": self._page_norm(),
                "step": self._step_norm(),
            }
            state = self._featurizer.build(self.persona, self.cognitive_state, self.intent.distribution.tolist(), memory_feats, quality, prior)
            gate = np.asarray([
                state.llm_entropy,
                quality,
                self._step_norm(),
                self._click_norm(),
                self.cognitive_state.frustration,
                self.cognitive_state.knowledge_state,
            ], dtype=np.float32)
            probs = self.calibrator.action_probs(state.vector(), np.log(prior).astype(np.float32), gate)
            probs = np.clip(probs, self.config.calibration.epsilon, None)
            probs /= probs.sum()
        chosen = CALIB_ACTIONS[int(np.random.choice(len(CALIB_ACTIONS), p=probs))]
        return chosen, dict(zip(CALIB_ACTIONS, map(float, probs)))

    @staticmethod
    def _persona_coord(persona: Persona, name: str) -> float:
        if name == "expertise":
            return {"novice": 0.0, "intermediate": 0.5, "expert": 1.0}.get(persona.search_expertise, 0.5)
        return {"impulsive": 0.0, "deliberate": 1.0, "price_sensitive": 0.5}.get(persona.decision_style, 0.5)

    def _rank_free_candidates(self, candidates):
        order = list(range(len(candidates)))
        key = f"{self.persona.persona_id}:{self.step_count}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        rng.shuffle(order)
        objects = []
        labels = []
        for out_idx, local_idx in enumerate(order):
            _, result = candidates[local_idx]
            label = f"C{out_idx}"
            title = str(getattr(result, "title", "") or "")
            snippet = str(getattr(result, "snippet", "") or "")
            url = str(getattr(result, "url", "") or "")
            masks = {
                "title": bool(getattr(result, "title_available", bool(title))),
                "snippet": bool(getattr(result, "snippet_available", bool(snippet))),
                "url": bool(getattr(result, "url_available", bool(url))),
            }
            text_blind = not any(masks.values())
            item = {
                "candidate_label": label,
                "title": title if masks["title"] else "",
                "snippet": snippet if masks["snippet"] else "",
                "url": url if masks["url"] else "",
                "field_masks": masks,
                "text_blind": text_blind,
            }
            opaque_document_id = str(getattr(result, "opaque_document_id", "") or "")
            opaque_domain_id = str(getattr(result, "opaque_domain_id", "") or "")
            if opaque_document_id:
                item["opaque_document_id"] = opaque_document_id
            if opaque_domain_id:
                item["opaque_domain_id"] = opaque_domain_id
            objects.append(item)
            labels.append(label)
        return order, objects, labels

    def select_click_target(self, serp_description: str, click_type: str, serp_results: Optional[list] = None) -> int:
        serp_results = list(serp_results or [])
        candidates = list(enumerate(serp_results))
        if not candidates:
            return 0
        if len(candidates) == 1:
            return candidates[0][0]
        return self._select_click_calibrated(candidates)

    def _select_click_calibrated(self, candidates) -> int:
        order, objects, labels = self._rank_free_candidates(candidates)
        text_blind = all(obj["text_blind"] for obj in objects)
        if text_blind:
            semantic = {label: 1.0 / len(labels) for label in labels}
        else:
            prompt = build_click_selection_prompt(self._persona_desc, self.current_query, objects)
            semantic = self.llm.get_action_logprobs(prompt, labels, rotation_key=f"select:{self.persona.persona_id}:{self.step_count}")
        semantic_by_local = np.zeros(len(candidates), dtype=np.float64)
        mtext_by_local = np.zeros(len(candidates), dtype=np.float64)
        for out_idx, local_idx in enumerate(order):
            semantic_by_local[local_idx] = semantic.get(labels[out_idx], 0.0)
            mtext_by_local[local_idx] = 0.0 if objects[out_idx]["text_blind"] else 1.0
        semantic_by_local = np.clip(semantic_by_local, self.config.calibration.epsilon, None)
        semantic_by_local /= semantic_by_local.sum()
        ranks = np.asarray([max(1, int(getattr(result, "rank", i + 1))) for i, (_, result) in enumerate(candidates)], dtype=np.float32)
        if self.calibrator is None:
            scores = semantic_by_local / ranks
            probs = scores / scores.sum()
        else:
            exam_context = np.asarray([
                self._page_norm(),
                self._step_norm(),
                self.cognitive_state.exploration_focus,
                self._persona_coord(self.persona, "expertise"),
                self._click_norm(),
            ], dtype=np.float32)
            style = self._persona_coord(self.persona, "style")
            attr = np.stack([
                np.log(semantic_by_local + self.config.calibration.epsilon),
                np.full(len(candidates), style),
                mtext_by_local,
            ], axis=1).astype(np.float32)
            probs = self.calibrator.click_probs(exam_context, attr, ranks)
            probs = np.clip(probs, self.config.calibration.epsilon, None)
            probs /= probs.sum()
        local = int(np.random.choice(len(candidates), p=probs))
        return candidates[local][0]

    def rate_satisfaction(self, clicked_title: str, clicked_snippet: str, is_ad: bool, clicked_url: str = "") -> float:
        prompt = build_satisfaction_prompt(
            self._persona_desc,
            clicked_title,
            clicked_snippet,
            is_ad,
            current_query=self.current_query,
            clicked_url=clicked_url,
        )
        probs = self.llm.get_action_logprobs(prompt, UTILITY_LABELS, rotation_key=f"utility:{self.persona.persona_id}:{self.step_count}")
        value = sum(i * probs.get(f"U{i}", 0.0) for i in range(1, 6))
        return float(np.clip(value, 1.0, 5.0))

    @staticmethod
    def estimate_dwell_time(satisfaction: float, is_ad: bool = False) -> float:
        bases = np.asarray([5.0, 15.0, 35.0, 70.0, 120.0], dtype=np.float64)
        s = float(np.clip(satisfaction, 1.0, 5.0))
        lo = int(np.floor(s))
        hi = int(np.ceil(s))
        if lo == hi:
            base = bases[lo - 1]
        else:
            w = s - lo
            base = (1.0 - w) * bases[lo - 1] + w * bases[hi - 1]
        return float(base * np.random.lognormal(mean=0.0, sigma=0.35))

    def generate_query(self, is_reformulation: bool = True) -> str:
        prompt = build_query_generation_prompt(
            persona_desc=self._persona_desc,
            original_query=self.current_query,
            memory_context=self.memory.get_recent_records(),
            intent_description=self.intent.distribution.tolist(),
            is_reformulation=is_reformulation,
        )
        return self.llm.generate_query(prompt, self.current_query)

    def record_action(self, action: str, target: str = "", result_title: str = "", is_ad: bool = False, reasoning: str = "", satisfaction: float = 0.0, dwell_time: float = 0.0) -> None:
        self.memory.add(MemoryEntry(
            step=self.step_count,
            query=self.current_query,
            action=action,
            target=target,
            result_title=result_title,
            is_ad=is_ad,
            satisfaction=satisfaction,
            dwell_time=dwell_time,
            reasoning=reasoning,
        ))
        self.intent.update(action)
        self.cognitive_state.update(action, satisfaction)
        if action in ("click_organic", "click_ad", "click_result"):
            self._total_clicks += 1
            self._last_satisfaction = satisfaction
        elif action == "skip":
            self._total_skips += 1
        self.step_count += 1

    def reset(self) -> None:
        self.memory.reset()
        self.intent.reset(self.config.intent.initial_distribution)
        self.cognitive_state.reset()
        self.current_query = self.persona.search_topic
        self.step_count = 0
        self._last_satisfaction = 0.0
        self._total_clicks = 0
        self._total_skips = 0
