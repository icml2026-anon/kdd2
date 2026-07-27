"""Ablation variants of SearchUserAgent.

Each variant disables one specific component to measure its contribution:
- A1: No persona bias (raw LLM probs, no trait-based adjustments)
- A2: No intent tracking (static intent distribution, never updated)
- A3: No memory (fresh context at each step)
- A4: No ad modeling (treat ads same as organic; remove ad-specific logic)
- A5: No position bias in click selection (uniform over candidates)
- A6: No cognitive search state (disable CSS module)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from searchagent.agent.llm_backbone import LLMBackbone
from searchagent.agent.memory import AgentMemory, MemoryEntry
from searchagent.agent.persona import Persona
from searchagent.agent.prompts import (
    ACTION_LABELS,
    MACRO_ACTIONS,
    build_action_selection_prompt,
    build_click_selection_prompt,
    build_query_generation_prompt,
)
from searchagent.agent.search_agent import CognitiveSearchState, IntentTracker, SearchUserAgent
from searchagent.config import IntentConfig, SearchAgentConfig

logger = logging.getLogger(__name__)


class AblationAgent(SearchUserAgent):
    """SearchUserAgent with configurable ablation flags.

    Args:
        disable_persona_bias: Skip _apply_persona_bias (A1).
        disable_intent: Never update intent distribution (A2).
        disable_memory: Always pass empty memory context (A3).
        disable_ad_model: Merge click_ad into click_organic (A4).
        disable_position_bias: Uniform click target distribution (A5).
    """

    def __init__(
        self,
        persona: Persona,
        llm: LLMBackbone,
        config: SearchAgentConfig,
        disable_persona_bias: bool = False,
        disable_intent: bool = False,
        disable_memory: bool = False,
        disable_ad_model: bool = False,
        disable_position_bias: bool = False,
        disable_css: bool = False,
    ):
        super().__init__(persona, llm, config)
        self.disable_persona_bias = disable_persona_bias
        self.disable_intent = disable_intent
        self.disable_memory = disable_memory
        self.disable_ad_model = disable_ad_model
        self.disable_position_bias = disable_position_bias
        self.disable_css = disable_css
        self._ablation_name = self._compute_name()

    def _compute_name(self) -> str:
        parts = []
        if self.disable_persona_bias:
            parts.append("no_persona")
        if self.disable_intent:
            parts.append("no_intent")
        if self.disable_memory:
            parts.append("no_memory")
        if self.disable_ad_model:
            parts.append("no_ad")
        if self.disable_position_bias:
            parts.append("no_posbias")
        if self.disable_css:
            parts.append("no_css")
        return "+".join(parts) if parts else "full_model"

    @property
    def ablation_name(self) -> str:
        return self._ablation_name

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        # Persona ablation: use generic description instead of trait-based one
        if self.disable_persona_bias:
            persona_desc = "A web search engine user."
        else:
            persona_desc = self._persona_desc

        if self.disable_ad_model:
            # Strip ad attitude from persona
            lines = [line for line in persona_desc.split("\n")
                     if not any(w in line.lower() for w in ["advertisement", "ads", "advertising"])]
            persona_desc = "\n".join(lines)
            # Sanitise SERP description: remove "Sponsored" mentions
            serp_description = serp_description.replace(" Sponsored", "").replace("Sponsored", "")

        # Memory ablation: suppress context
        if self.disable_memory:
            memory_ctx = "No previous actions in this session."
        else:
            memory_ctx = self.memory.get_recent_context()

        # Intent ablation: use static description
        if self.disable_intent:
            intent_desc = "Intent: general search"
        else:
            intent_desc = self.intent.describe()

        # CSS ablation: suppress cognitive state context
        if self.disable_css:
            css_ctx = ""
        else:
            css_ctx = self.cognitive_state.describe()

        prompt = build_action_selection_prompt(
            persona_desc=persona_desc,
            current_query=self.current_query,
            serp_description=serp_description,
            memory_context=memory_ctx,
            intent_description=intent_desc,
            css_context=css_ctx,
        )

        action_probs = self.llm.get_action_logprobs(prompt, ACTION_LABELS)
        named_probs = {}
        for label, action in zip(ACTION_LABELS, MACRO_ACTIONS):
            named_probs[action] = action_probs.get(label, 0.0)

        # Persona bias ablation: skip adjustment
        if not self.disable_persona_bias:
            named_probs = self._apply_persona_bias(named_probs)

        # CSS module owns: cognitive modifiers
        if not self.disable_css:
            named_probs = self._apply_css_modifiers(named_probs)
        # Satisfaction feedback is a universal session dynamic (always applied, like fatigue)
        named_probs = self._apply_satisfaction_feedback(named_probs)

        # Intent module owns: intent-driven action gating
        named_probs = self._apply_intent_modifiers(named_probs)

        # Memory module owns: click satiation (memory of past clicks)
        if not self.disable_memory:
            named_probs = self._apply_click_satiation(named_probs)

        # Session fatigue: universal baseline (always applied)
        named_probs = self._apply_session_fatigue(named_probs)

        actions = list(named_probs.keys())
        probs = np.array([named_probs[a] for a in actions])
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()

        chosen_idx = np.random.choice(len(actions), p=probs)
        chosen_action = actions[chosen_idx]

        return chosen_action, named_probs

    def _get_ad_weight(self, result) -> float:
        if self.disable_ad_model or self.disable_persona_bias:
            return 1.0
        return super()._get_ad_weight(result)

    def _apply_persona_bias(self, probs: Dict[str, float]) -> Dict[str, float]:
        if self.disable_persona_bias:
            return probs
        adjusted = super()._apply_persona_bias(probs)
        if self.disable_ad_model:
            # Under disable_ad_model, we neutralize ad_sensitivity bias if click_ad is used
            if "click_ad" in adjusted:
                sens = self.persona.ad_sensitivity
                if sens == "ad_averse":
                    adjusted["click_ad"] /= 0.3
                elif sens == "ad_tolerant":
                    adjusted["click_ad"] /= 1.5
                total = sum(adjusted.values())
                if total > 0:
                    adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted

    def select_click_target(
        self, serp_description: str, click_type: str,
        serp_results: Optional[list] = None,
    ) -> int:
        if self.disable_ad_model:
            # Sanitise SERP description: remove "Sponsored" mentions
            serp_description = serp_description.replace(" Sponsored", "").replace("Sponsored", "")

        persona_desc = self._persona_desc
        if self.disable_persona_bias:
            persona_desc = "A web search engine user."
        elif self.disable_ad_model:
            # Strip ad attitude from persona
            lines = [line for line in persona_desc.split("\n")
                     if not any(w in line.lower() for w in ["advertisement", "ads", "advertising"])]
            persona_desc = "\n".join(lines)

        memory_ctx = self.memory.get_recent_context()
        if self.disable_memory:
            memory_ctx = "No previous actions in this session."

        if self.disable_position_bias:
            if not serp_results:
                return 0
            if click_type == "click_ad":
                candidates = [(i, r) for i, r in enumerate(serp_results) if r.is_ad]
            else:
                candidates = list(enumerate(serp_results))
            if not candidates:
                candidates = list(enumerate(serp_results))
            ad_w = np.array([self._get_ad_weight(r) for _, r in candidates])
            ad_w = np.clip(ad_w, 1e-8, None)
            ad_w /= ad_w.sum()
            chosen_local = np.random.choice(len(candidates), p=ad_w)
            return candidates[chosen_local][0]

        # Use overridden variables
        saved_persona = self._persona_desc
        saved_memory = self.memory
        self._persona_desc = persona_desc
        if self.disable_memory:
            self.memory = AgentMemory()
        try:
            result = super().select_click_target(serp_description, click_type, serp_results)
        finally:
            self._persona_desc = saved_persona
            self.memory = saved_memory
        return result

    def rate_satisfaction(self, clicked_title: str, clicked_snippet: str, is_ad: bool) -> float:
        persona_desc = self._persona_desc
        if self.disable_persona_bias:
            persona_desc = "A web search engine user."
        elif self.disable_ad_model:
            lines = [line for line in persona_desc.split("\n")
                     if not any(w in line.lower() for w in ["advertisement", "ads", "advertising"])]
            persona_desc = "\n".join(lines)
            is_ad = False  # Completely blind to ad status in satisfaction evaluation

        saved_persona = self._persona_desc
        self._persona_desc = persona_desc
        try:
            result = super().rate_satisfaction(clicked_title, clicked_snippet, is_ad)
        finally:
            self._persona_desc = saved_persona
        return result

    def generate_query(self, is_reformulation: bool = True) -> str:
        persona_desc = self._persona_desc
        if self.disable_persona_bias:
            persona_desc = "A web search engine user."
        elif self.disable_ad_model:
            lines = [line for line in persona_desc.split("\n")
                     if not any(w in line.lower() for w in ["advertisement", "ads", "advertising"])]
            persona_desc = "\n".join(lines)

        memory_ctx = self.memory.get_recent_context()
        if self.disable_memory:
            memory_ctx = "No previous actions in this session."

        saved_persona = self._persona_desc
        saved_memory = self.memory
        self._persona_desc = persona_desc
        if self.disable_memory:
            self.memory = AgentMemory()
        try:
            result = super().generate_query(is_reformulation)
        finally:
            self._persona_desc = saved_persona
            self.memory = saved_memory
        return result

    def record_action(
        self, action: str, target: str = "", result_title: str = "",
        is_ad: bool = False, reasoning: str = "", satisfaction: float = 0.0,
    ) -> None:
        super().record_action(action, target, result_title, is_ad, reasoning, satisfaction)
        # Intent ablation: undo the intent update that super() did
        if self.disable_intent:
            self.intent.reset(self.config.intent.initial_distribution)
        # CSS ablation: keep CSS frozen at initial prior values
        if self.disable_css:
            self.cognitive_state.reset()


# Predefined ablation configurations
ABLATION_VARIANTS = {
    "full": {},
    "A1_no_persona": {"disable_persona_bias": True},
    "A2_no_intent": {"disable_intent": True},
    "A3_no_memory": {"disable_memory": True},
    "A4_no_ad": {"disable_ad_model": True},
    "A5_no_posbias": {"disable_position_bias": True},
    "A6_no_css": {"disable_css": True},
}


def create_ablation_agent(
    variant: str,
    persona: Persona,
    llm: LLMBackbone,
    config: SearchAgentConfig,
) -> AblationAgent:
    """Factory function to create an ablation agent by variant name."""
    if variant not in ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown ablation variant '{variant}'. "
            f"Available: {list(ABLATION_VARIANTS.keys())}"
        )
    kwargs = ABLATION_VARIANTS[variant]
    return AblationAgent(persona=persona, llm=llm, config=config, **kwargs)
