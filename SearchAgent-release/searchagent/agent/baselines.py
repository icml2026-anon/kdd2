"""Baseline agents for comparison experiments.

Three baselines:
1. RandomAgent: Uniformly random action selection
2. RuleBasedAgent: Position-biased clicks with fixed behavioral rules
3. NaiveLLMAgent: LLM-driven but WITHOUT persona bias, intent tracking, or memory
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from searchagent.agent.llm_backbone import LLMBackbone
from searchagent.agent.memory import AgentMemory, MemoryEntry
from searchagent.agent.persona import Persona
from searchagent.agent.prompts import (
    ACTION_LABELS,
    MACRO_ACTIONS,
)
from searchagent.config import SearchAgentConfig

logger = logging.getLogger(__name__)


class RandomAgent:
    """Baseline: uniformly random action selection and random click targets.

    No LLM calls — purely stochastic. Serves as the lower bound
    for behavioral fidelity.
    """

    def __init__(self, persona: Persona, config: SearchAgentConfig):
        self.persona = persona
        self.config = config
        self.current_query = persona.search_topic
        self.step_count = 0
        self.memory = AgentMemory()
        # Fixed action distribution (slightly weighted toward realistic behavior)
        if len(MACRO_ACTIONS) == 7:
            self._action_probs = np.array([
                0.29,  # click_result
                0.18,  # skip
                0.06,  # next_page
                0.10,  # reformulate
                0.07,  # new_query
                0.20,  # stop_satisfied
                0.10,  # stop_unsatisfied
            ])
        else:
            self._action_probs = np.array([
                0.25,  # click_organic
                0.04,  # click_ad
                0.18,  # skip
                0.06,  # next_page
                0.10,  # reformulate
                0.07,  # new_query
                0.10,  # abandon
                0.20,  # end_session
            ])
        self._action_probs /= self._action_probs.sum()

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        probs = dict(zip(MACRO_ACTIONS, self._action_probs))
        chosen = np.random.choice(MACRO_ACTIONS, p=self._action_probs)
        return chosen, probs

    def select_click_target(
        self, serp_description: str, click_type: str,
        serp_results: Optional[list] = None,
    ) -> int:
        if not serp_results:
            return 0
        if click_type == "click_ad":
            candidates = [i for i, r in enumerate(serp_results) if r.is_ad]
        elif click_type == "click_organic":
            candidates = [i for i, r in enumerate(serp_results) if not r.is_ad]
        else:
            candidates = list(range(len(serp_results)))
        if not candidates:
            candidates = list(range(len(serp_results)))
        return int(np.random.choice(candidates))

    def generate_query(self, is_reformulation: bool = True) -> str:
        # Random agent can't generate queries — append a suffix
        suffix = np.random.choice(["review", "comparison", "best", "cheap", "2024"])
        if is_reformulation:
            return f"{self.current_query} {suffix}"
        return f"{self.persona.search_topic} {suffix}"

    def record_action(self, action: str, target: str = "", result_title: str = "",
                      is_ad: bool = False, reasoning: str = "",
                      satisfaction: float = 0.0) -> None:
        entry = MemoryEntry(
            step=self.step_count, query=self.current_query,
            action=action, target=target, result_title=result_title,
            is_ad=is_ad, reasoning=reasoning,
        )
        self.memory.add(entry)
        self.step_count += 1

    def reset(self) -> None:
        self.memory.reset()
        self.current_query = self.persona.search_topic
        self.step_count = 0


class RuleBasedAgent:
    """Baseline: deterministic rules based on position bias and click models.

    Implements a simplified cascade click model:
    - Examine results top-to-bottom
    - Click with probability proportional to position bias × relevance
    - Reformulate if no click after scanning the page
    - Abandon after N unsuccessful pages

    No LLM calls. Deterministic rules with stochastic click decisions.
    """

    def __init__(self, persona: Persona, config: SearchAgentConfig):  # noqa: E501
        self.persona = persona
        self.config = config
        self.current_query = persona.search_topic
        self.step_count = 0
        self.memory = AgentMemory()
        self.pos_bias = np.array(config.environment.position_bias_click_prob)
        self._scan_position = 0
        self._pages_without_click = 0
        self._clicks_this_page = 0
        self._max_pages_before_abandon = 3
        self._reformulate_after_scans = 5
        self._scans_without_click = 0

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        # Rule-based decision tree
        probs = {a: 0.0 for a in MACRO_ACTIONS}

        abandon_key = "stop_unsatisfied" if "stop_unsatisfied" in MACRO_ACTIONS else "abandon"
        end_key = "stop_satisfied" if "stop_satisfied" in MACRO_ACTIONS else "end_session"
        click_key = "click_result" if "click_result" in MACRO_ACTIONS else "click_organic"

        # After too many pages without clicks → abandon
        if self._pages_without_click >= self._max_pages_before_abandon:
            probs[abandon_key] = 1.0
            return abandon_key, probs

        # After scanning several results without click → reformulate
        if self._scans_without_click >= self._reformulate_after_scans:
            self._scans_without_click = 0
            probs["reformulate"] = 1.0
            return "reformulate", probs

        # After 1-2 clicks → likely satisfied, end session
        if self._clicks_this_page >= 1 and np.random.random() < 0.4:
            probs[end_key] = 1.0
            return end_key, probs

        # After 2 clicks on same page → next page or reformulate
        if self._clicks_this_page >= 2:
            self._clicks_this_page = 0
            action = np.random.choice(["next_page", end_key], p=[0.4, 0.6])
            probs[action] = 1.0
            return action, probs

        # Main decision: click with position-biased probability
        rank = min(self._scan_position, len(self.pos_bias) - 1)
        click_prob = self.pos_bias[rank]

        # Ad click probability is lower (rule-based ad aversion)
        ad_click_prob = click_prob * 0.3

        r = np.random.random()
        if r < click_prob:
            if click_key == "click_result":
                probs["click_result"] = click_prob
                return "click_result", probs
            else:
                probs["click_organic"] = click_prob
                probs["click_ad"] = ad_click_prob
                # Decide organic vs ad based on what's available
                if np.random.random() < 0.1:  # 10% chance it's an ad click
                    return "click_ad", probs
                return "click_organic", probs
        elif r < click_prob + 0.15:
            probs["skip"] = 1.0
            self._scan_position += 1
            self._scans_without_click += 1
            return "skip", probs
        else:
            probs["skip"] = 1.0
            self._scan_position += 1
            self._scans_without_click += 1
            return "skip", probs

    def select_click_target(
        self, serp_description: str, click_type: str,
        serp_results: Optional[list] = None,
    ) -> int:
        """Position-biased click selection using cascade model."""
        if not serp_results:
            return 0

        if click_type == "click_ad":
            candidates = [(i, r) for i, r in enumerate(serp_results) if r.is_ad]
        elif click_type == "click_organic":
            candidates = [(i, r) for i, r in enumerate(serp_results) if not r.is_ad]
        else:
            candidates = list(enumerate(serp_results))

        if not candidates:
            candidates = list(enumerate(serp_results))

        # Weight by position bias
        weights = []
        for idx, r in candidates:
            rank_idx = min(r.rank - 1, len(self.pos_bias) - 1)
            weights.append(self.pos_bias[rank_idx] * r.relevance)

        weights = np.array(weights)
        weights = np.clip(weights, 1e-8, None)
        weights /= weights.sum()

        chosen_local = np.random.choice(len(candidates), p=weights)
        self._clicks_this_page += 1
        self._scans_without_click = 0
        self._pages_without_click = 0
        return candidates[chosen_local][0]

    def generate_query(self, is_reformulation: bool = True) -> str:
        suffixes = ["review", "comparison", "best deal", "reddit", "2024 guide"]
        suffix = np.random.choice(suffixes)
        if is_reformulation:
            return f"{self.current_query} {suffix}"
        return f"{self.persona.search_topic} {suffix}"

    def record_action(self, action: str, target: str = "", result_title: str = "",
                      is_ad: bool = False, reasoning: str = "",
                      satisfaction: float = 0.0) -> None:
        entry = MemoryEntry(
            step=self.step_count, query=self.current_query,
            action=action, target=target, result_title=result_title,
            is_ad=is_ad, reasoning=reasoning,
        )
        self.memory.add(entry)
        self.step_count += 1
        if action == "next_page":
            self._clicks_this_page = 0
            self._scan_position = 0
            self._pages_without_click += 1
        elif action in ("reformulate", "new_query"):
            self._clicks_this_page = 0
            self._scan_position = 0
            self._pages_without_click = 0

    def reset(self) -> None:
        self.memory.reset()
        self.current_query = self.persona.search_topic
        self.step_count = 0
        self._scan_position = 0
        self._pages_without_click = 0
        self._clicks_this_page = 0
        self._scans_without_click = 0


class NaiveLLMAgent:
    """Baseline: LLM-driven but stripped of SearchAgent's key innovations.

    Uses the LLM for action selection but:
    - NO persona bias adjustments
    - NO intent tracking (static prompt)
    - NO memory context (each step is independent)
    - NO position bias in click selection

    This isolates the contribution of our persona/intent/memory modules.
    """

    def __init__(self, persona: Persona, llm: LLMBackbone, config: SearchAgentConfig):
        self.persona = persona
        self.llm = llm
        self.config = config
        self.current_query = persona.search_topic
        self.step_count = 0
        self.memory = AgentMemory()
        self._total_clicks = 0
        self._last_satisfaction = 0.0
        # Minimal persona description (just the search topic)
        self._persona_desc = f"You are searching for: {persona.search_topic}"

    def _apply_session_dynamics(self, named_probs: Dict[str, float]) -> Dict[str, float]:
        """Apply basic session fatigue and click satiation (no persona/intent)."""
        adjusted = dict(named_probs)

        # Click satiation (TREC 2014: mean 1.34 clicks/session)
        if self._total_clicks >= 1:
            satiation = min(self._total_clicks, 5)
            for k in ("stop_satisfied", "end_session"):
                if k in adjusted:
                    adjusted[k] = adjusted.get(k, 0.05) * (2.0 ** satiation)
            dampen = max(0.05, 0.4 ** satiation)
            for k in ("click_result", "click_organic", "click_ad"):
                if k in adjusted:
                    adjusted[k] *= dampen
            if self._total_clicks >= 2:
                for k in ("reformulate", "abandon", "stop_unsatisfied"):
                    if k in adjusted:
                        adjusted[k] *= 1.5

        # Session fatigue (reaches max at step 3)
        fatigue = min(1.0, self.step_count / 3.0)
        boost = 1.0 + fatigue * 7.0
        for k in ("stop_satisfied", "end_session"):
            if k in adjusted:
                adjusted[k] = adjusted.get(k, 0.05) * boost
        for k in ("stop_unsatisfied", "abandon"):
            if k in adjusted:
                adjusted[k] *= (1.0 + fatigue * 3.0)
        dampen = max(0.1, 1.0 - fatigue * 0.8)
        for a in ["click_result", "click_organic", "click_ad", "skip", "next_page"]:
            if a in adjusted:
                adjusted[a] *= dampen

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        from searchagent.agent.prompts import build_action_selection_prompt
        prompt = build_action_selection_prompt(
            persona_desc=self._persona_desc,
            current_query=self.current_query,
            serp_description=serp_description,
            memory_context="No previous actions.",  # no memory
            intent_description="General search",  # no intent tracking
        )
        action_probs = self.llm.get_action_logprobs(prompt, ACTION_LABELS)

        named_probs = {}
        for label, action in zip(ACTION_LABELS, MACRO_ACTIONS):
            named_probs[action] = action_probs.get(label, 0.0)

        # NO persona bias applied — raw LLM probs
        # But apply basic session dynamics (fatigue + satiation)
        named_probs = self._apply_session_dynamics(named_probs)

        actions = list(named_probs.keys())
        probs = np.array([named_probs[a] for a in actions])
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()

        chosen_idx = np.random.choice(len(actions), p=probs)
        return actions[chosen_idx], named_probs

    def select_click_target(
        self, serp_description: str, click_type: str,
        serp_results: Optional[list] = None,
    ) -> int:
        """Naive click selection: just pick the first matching result."""
        if not serp_results:
            return 0

        if click_type == "click_ad":
            for i, r in enumerate(serp_results):
                if r.is_ad:
                    return i
        elif click_type == "click_organic":
            for i, r in enumerate(serp_results):
                if not r.is_ad:
                    return i
        # Default fallback / click_result
        return 0

    def generate_query(self, is_reformulation: bool = True) -> str:
        from searchagent.agent.prompts import build_query_generation_prompt
        prompt = build_query_generation_prompt(
            persona_desc=self._persona_desc,
            original_query=self.current_query,
            memory_context="No previous actions.",
            intent_description="General search",
            is_reformulation=is_reformulation,
        )
        new_query = self.llm.generate_text(prompt, max_new_tokens=32)
        new_query = new_query.split("\n")[0].strip().strip('"').strip("'")
        return new_query if new_query else self.current_query

    def record_action(self, action: str, target: str = "", result_title: str = "",
                      is_ad: bool = False, reasoning: str = "",
                      satisfaction: float = 0.0) -> None:
        entry = MemoryEntry(
            step=self.step_count, query=self.current_query,
            action=action, target=target, result_title=result_title,
            is_ad=is_ad, reasoning=reasoning,
        )
        self.memory.add(entry)
        if action in ("click_organic", "click_ad", "click_result"):
            self._total_clicks += 1
            self._last_satisfaction = satisfaction
        self.step_count += 1

    def reset(self) -> None:
        self.memory.reset()
        self.current_query = self.persona.search_topic
        self.step_count = 0
        self._total_clicks = 0
        self._last_satisfaction = 0.0


class DualPromptLLMAgent(NaiveLLMAgent):
    """Baseline: Dual-prompt LLM User Simulator (inspired by WSDM 2023 UserSimCRS).

    Separates the decision into two distinct steps:
    1. Stop/Continue decision (Prompt 1).
    2. Specific Action selection if continuing (Prompt 2).
    """

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        # Prompt 1: Decide Stop or Continue
        stop_prompt = (
            f"You are searching for: {self.current_query}.\n"
            f"The search results page (SERP) is described as follows:\n{serp_description}\n"
            f"Total clicks in session: {self._total_clicks}. Last click satisfaction: {self._last_satisfaction}.\n\n"
            f"Should you stop searching now? Reply with 'TRUE' to stop, or 'FALSE' to continue searching."
        )
        try:
            stop_response = self.llm.generate_text(stop_prompt, max_new_tokens=5).strip().upper()
            should_stop = "TRUE" in stop_response or "YES" in stop_response
        except Exception:
            should_stop = False

        probs = {a: 0.0 for a in MACRO_ACTIONS}

        abandon_key = "stop_unsatisfied" if "stop_unsatisfied" in MACRO_ACTIONS else "abandon"
        end_key = "stop_satisfied" if "stop_satisfied" in MACRO_ACTIONS else "end_session"

        if should_stop:
            # If stopping, decide based on satisfaction
            chosen_action = end_key if self._last_satisfaction >= 3.0 else abandon_key
            probs[chosen_action] = 1.0
            return chosen_action, probs

        # Prompt 2: If continuing, decide next specific action
        from searchagent.agent.prompts import build_action_selection_prompt
        prompt2 = build_action_selection_prompt(
            persona_desc=self._persona_desc,
            current_query=self.current_query,
            serp_description=serp_description,
            memory_context="No previous actions.",
            intent_description="General search",
        )
        action_probs = self.llm.get_action_logprobs(prompt2, ACTION_LABELS)

        named_probs = {}
        for label, action in zip(ACTION_LABELS, MACRO_ACTIONS):
            named_probs[action] = action_probs.get(label, 0.0)

        # Force stopping actions to be 0 since Prompt 1 decided to CONTINUE
        for k in (end_key, abandon_key):
            if k in named_probs:
                named_probs[k] = 0.0

        # Apply basic session dynamics (fatigue + satiation)
        named_probs = self._apply_session_dynamics(named_probs)

        actions = list(named_probs.keys())
        probs_array = np.array([named_probs[a] for a in actions])
        probs_array = np.clip(probs_array, 1e-8, None)
        probs_array /= probs_array.sum()

        chosen_idx = np.random.choice(len(actions), p=probs_array)
        return actions[chosen_idx], named_probs


class UXSimHybridAgent(RuleBasedAgent):
    """Baseline: Hybrid User Simulator (inspired by UXSim CIKM 2025).

    Utilizes rule-based decision logic for click and action choices (Cascade Click Model),
    but invokes the LLM to generate realistic query reformulations instead of simple suffixes.
    """

    def __init__(self, persona: Persona, llm: LLMBackbone, config: SearchAgentConfig):
        super().__init__(persona, config)
        self.llm = llm
        self._persona_desc = f"You are searching for: {persona.search_topic}"

    def generate_query(self, is_reformulation: bool = True) -> str:
        from searchagent.agent.prompts import build_query_generation_prompt
        prompt = build_query_generation_prompt(
            persona_desc=self._persona_desc,
            original_query=self.current_query,
            memory_context="No previous actions.",
            intent_description="General search",
            is_reformulation=is_reformulation,
        )
        try:
            new_query = self.llm.generate_text(prompt, max_new_tokens=32)
            new_query = new_query.split("\n")[0].strip().strip('"').strip("'")
            return new_query if new_query else self.current_query
        except Exception:
            return super().generate_query(is_reformulation)


class USimAgent(NaiveLLMAgent):
    """Baseline: USimAgent — ReAct-style iterative search-session simulator (SIGIR 2024).

    Reference: "USimAgent: Large Language Models for Simulating Search Users"
    (Zhang et al., SIGIR 2024). The agent runs a per-round loop that interleaves
    explicit reasoning with the search actions (reason -> decide stop -> act),
    and crucially accumulates the rolling session context across rounds rather
    than dumping a static profile up front. Distinct from a plain persona+history
    prompt: the LLM first emits a free-text reasoning trace conditioned on the
    full accumulated session, and that trace is fed back as context for the
    next-round decision.
    """

    def __init__(self, persona: Persona, llm: LLMBackbone, config: SearchAgentConfig):
        super().__init__(persona, llm, config)
        self._persona_desc = (
            f"You are searching for: {persona.search_topic}.\n"
            f"Your traits:\n"
            f"- Patience Level: {persona.patience_level}\n"
            f"- Search Expertise: {persona.search_expertise}\n"
            f"- Ad Sensitivity: {persona.ad_sensitivity}\n"
            f"- Decision Style: {persona.decision_style}"
        )
        self._reasoning_trace = ""  # rolling reasoning accumulated across rounds

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        # Step 1: accumulate session context (history + prior reasoning trace)
        history_lines = []
        for entry in self.memory.entries:
            history_lines.append(
                f"Step {entry.step}: Query '{entry.query}', Action '{entry.action}', Target '{entry.target}'")
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."

        # Step 2: emit a reasoning trace conditioned on the accumulated session
        try:
            reason_prompt = (
                f"You simulate a human web-search user.\n{self._persona_desc}\n\n"
                f"Current query: {self.current_query}\n"
                f"Search results:\n{serp_description}\n\n"
                f"Session so far:\n{memory_context}\n"
                f"Prior reasoning:\n{self._reasoning_trace or '(none)'}\n\n"
                f"In 1-2 sentences, reason about what this user would do next and why."
            )
            trace = self.llm.generate_text(reason_prompt, max_new_tokens=80).strip()
            if trace:
                self._reasoning_trace = trace
        except Exception:
            pass

        # Step 3: forced-choice action conditioned on the reasoning trace
        from searchagent.agent.prompts import build_action_selection_prompt
        prompt = build_action_selection_prompt(
            persona_desc=self._persona_desc + f"\nYour reasoning: {self._reasoning_trace}",
            current_query=self.current_query,
            serp_description=serp_description,
            memory_context=memory_context,
            intent_description=f"Active pursuit of: {self.persona.search_topic}",
        )
        action_probs = self.llm.get_action_logprobs(prompt, ACTION_LABELS)
        named_probs = {}
        for label, action in zip(ACTION_LABELS, MACRO_ACTIONS):
            named_probs[action] = action_probs.get(label, 0.0)
        named_probs = self._apply_session_dynamics(named_probs)

        actions = list(named_probs.keys())
        probs = np.array([named_probs[a] for a in actions])
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()
        chosen_idx = np.random.choice(len(actions), p=probs)
        return actions[chosen_idx], named_probs

    def reset(self) -> None:
        super().reset()
        self._reasoning_trace = ""



class ReActSearchAgent(NaiveLLMAgent):
    """Baseline: ReAct User Simulator (inspired by ReAct Yao et al. ICLR 23).

    Instructs the LLM to output a explicit "Thought" explaining its current cognitive
    reasoning before selecting the "Action", separating reasoning from execution.
    """

    def __init__(self, persona: Persona, llm: LLMBackbone, config: SearchAgentConfig):
        super().__init__(persona, llm, config)
        self._persona_desc = (
            f"You are searching for: {persona.search_topic}.\n"
            f"Your traits:\n"
            f"- Patience Level: {persona.patience_level}\n"
            f"- Search Expertise: {persona.search_expertise}\n"
            f"- Ad Sensitivity: {persona.ad_sensitivity}\n"
            f"- Decision Style: {persona.decision_style}"
        )

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        # Format history
        history_lines = []
        for entry in self.memory.entries:
            history_lines.append(f"Step {entry.step}: Query '{entry.query}', Action '{entry.action}', Target '{entry.target}'")
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."

        # Construct explicit ReAct Prompt
        react_prompt = (
            f"You are an agent simulating a human user searching the web.\n"
            f"Persona context:\n{self._persona_desc}\n\n"
            f"Search topic: {self.current_query}\n"
            f"SERP results:\n{serp_description}\n\n"
            f"Previous action history:\n{memory_context}\n\n"
            f"Allowed actions: {list(MACRO_ACTIONS)}\n\n"
            f"Use the ReAct format to decide your next step. You must strictly output in the following format:\n"
            f"Thought: [Explain your step-by-step reasoning about the search results and your cognitive goals]\n"
            f"Action: [Choose exactly one action from Allowed actions]\n"
        )

        try:
            raw_response = self.llm.generate_text(react_prompt, max_new_tokens=150)
            # Parse thought and action
            thought, action = "", ""
            for line in raw_response.split("\n"):
                if line.strip().startswith("Thought:"):
                    thought = line.replace("Thought:", "").strip()
                elif line.strip().startswith("Action:"):
                    action = line.replace("Action:", "").strip().lower()

            # Clean action string to match MACRO_ACTIONS
            chosen_action = "skip"  # default fallback
            for a in MACRO_ACTIONS:
                if a in action or action in a:
                    chosen_action = a
                    break
        except Exception:
            chosen_action = "skip"

        probs = {a: 0.0 for a in MACRO_ACTIONS}
        probs[chosen_action] = 1.0

        return chosen_action, probs

    def generate_query(self, is_reformulation: bool = True) -> str:
        # ReAct agent generates query using standard query prompt
        return super().generate_query(is_reformulation)


class SkillsBasedAgent(NaiveLLMAgent):
    """Baseline: Skills-Based User Simulator (inspired by Sim4IA SIGIR 2025).

    Decomposes the interactive search process into discrete, modular cognitive skills
    (Querying, Assessment, Clicking, and Termination) using independent prompt evaluations.
    """

    def termination_skill(self, serp_description: str) -> bool:
        """Modular Skill 1: Decide whether to end search or continue."""
        prompt = (
            f"You are simulating a search user. Topic: '{self.current_query}'\n"
            f"SERP context:\n{serp_description}\n"
            f"Step count: {self.step_count}, Total clicks: {self._total_clicks}.\n"
            f"Decide if you should STOP searching now. Answer strictly with 'YES' or 'NO'."
        )
        try:
            res = self.llm.generate_text(prompt, max_new_tokens=5).strip().upper()
            return "YES" in res or "TRUE" in res
        except Exception:
            return self.step_count >= 4

    def assessment_skill(self, serp_description: str) -> Optional[int]:
        """Modular Skill 2: Assess SERP results and identify relevant targets to click."""
        prompt = (
            f"Review this SERP description:\n{serp_description}\n\n"
            f"Is there a highly relevant result that matches your topic '{self.current_query}'?\n"
            f"If YES, output the rank index (e.g. 0, 1, 2) of the result. If NO relevant result is found, output 'NONE'."
        )
        try:
            res = self.llm.generate_text(prompt, max_new_tokens=5).strip().upper()
            if "NONE" in res:
                return None
            import re
            numbers = re.findall(r"\d+", res)
            return int(numbers[0]) if numbers else None
        except Exception:
            return None

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        # 1. Evaluate Termination Skill
        should_stop = self.termination_skill(serp_description)
        probs = {a: 0.0 for a in MACRO_ACTIONS}

        abandon_key = "stop_unsatisfied" if "stop_unsatisfied" in MACRO_ACTIONS else "abandon"
        end_key = "stop_satisfied" if "stop_satisfied" in MACRO_ACTIONS else "end_session"

        if should_stop:
            chosen = end_key if self._last_satisfaction >= 3.0 else abandon_key
            probs[chosen] = 1.0
            return chosen, probs

        # 2. Evaluate Assessment Skill
        target_rank = self.assessment_skill(serp_description)
        if target_rank is not None:
            # Decide to click organic (most common)
            click_key = "click_result" if "click_result" in MACRO_ACTIONS else "click_organic"
            probs[click_key] = 1.0
            # Cache the choice in a temporary field for select_click_target
            self._skills_target = target_rank
            return click_key, probs

        # 3. Fallback skills (Querying/Paging/Skipping)
        fallback_action = "skip"
        if np.random.random() < 0.4:
            fallback_action = "reformulate"
        elif np.random.random() < 0.15:
            fallback_action = "next_page"

        probs[fallback_action] = 1.0
        return fallback_action, probs

    def select_click_target(
        self, serp_description: str, click_type: str,
        serp_results: Optional[list] = None,
    ) -> int:
        # Use target chosen by assessment_skill if available and valid
        if hasattr(self, "_skills_target") and self._skills_target is not None:
            target = self._skills_target
            delattr(self, "_skills_target")
            if serp_results and 0 <= target < len(serp_results):
                return target
        return super().select_click_target(serp_description, click_type, serp_results)


class FewShotDemoAgent(NaiveLLMAgent):
    """Baseline: Few-Shot Demonstration Agent (inspired by UserSimCRS v2 ECIR 2026).

    Guids the LLM behavior by injecting realistic, annotated in-context search session
    demonstrations of similar user personas.
    """

    def __init__(self, persona: Persona, llm: LLMBackbone, config: SearchAgentConfig):
        super().__init__(persona, llm, config)
        self._persona_desc = (
            f"You are searching for: {persona.search_topic}.\n"
            f"Your traits:\n"
            f"- Patience Level: {persona.patience_level}\n"
            f"- Search Expertise: {persona.search_expertise}\n"
            f"- Ad Sensitivity: {persona.ad_sensitivity}\n"
            f"- Decision Style: {persona.decision_style}"
        )
        # Inject standard annotated session demonstration into prompt
        self._few_shot_context = (
            "--- START OF FEW-SHOT SEARCH DEMONSTRATION ---\n"
            "Topic: 'best budget camera 2024'\n"
            "Goal: Find a highly-rated, affordable digital camera under $500.\n"
            "Step 0: Query formulated: 'best budget camera 2024'\n"
            "Step 1: Scanned results page.\n"
            "Action Chosen: click_organic (Target rank: 0, Title: 'Best Budget Cameras of 2024 - CNET')\n"
            "Step 2: Read page. Satisfaction: 4.0. Returned to search results.\n"
            "Action Chosen: click_organic (Target rank: 1, Title: 'Cheap Cameras for Beginners under $500')\n"
            "Step 3: Read page. Satisfaction: 3.5. Found all info.\n"
            "Action Chosen: stop_satisfied\n"
            "--- END OF FEW-SHOT SEARCH DEMONSTRATION ---\n\n"
        )

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        # Format history
        history_lines = []
        for entry in self.memory.entries:
            history_lines.append(f"Step {entry.step}: Query '{entry.query}', Action '{entry.action}', Target '{entry.target}'")
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."

        # In-Context Learning Few-Shot Action Prompt
        prompt = (
            f"You are an agent simulating a search user. Follow the style of the demonstration below:\n\n"
            f"{self._few_shot_context}"
            f"Now, simulate your own search process.\n"
            f"Persona traits:\n{self._persona_desc}\n"
            f"Search topic: {self.current_query}\n"
            f"Current SERP:\n{serp_description}\n"
            f"Your active history:\n{memory_context}\n\n"
            f"Select your next action from allowed list: {list(MACRO_ACTIONS)}.\n"
            f"Allowed Action probabilities should align with your traits and history.\n"
        )
        action_probs = self.llm.get_action_logprobs(prompt, ACTION_LABELS)

        named_probs = {}
        for label, action in zip(ACTION_LABELS, MACRO_ACTIONS):
            named_probs[action] = action_probs.get(label, 0.0)

        # Apply basic session dynamics
        named_probs = self._apply_session_dynamics(named_probs)

        actions = list(named_probs.keys())
        probs = np.array([named_probs[a] for a in actions])
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()

        chosen_idx = np.random.choice(len(actions), p=probs)
        return actions[chosen_idx], named_probs

    def generate_query(self, is_reformulation: bool = True) -> str:
        # Few-Shot query prompt
        history_lines = []
        for entry in self.memory.entries:
            history_lines.append(f"Step {entry.step}: Query '{entry.query}', Action '{entry.action}', Target '{entry.target}'")
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."

        prompt = (
            f"Follow the style of the query generation in the demonstration:\n\n"
            f"{self._few_shot_context}"
            f"Current topic: {self.current_query}\n"
            f"Your active history:\n{memory_context}\n"
            f"Generate your next search query. Answer with the query text ONLY."
        )
        new_query = self.llm.generate_text(prompt, max_new_tokens=32)
        new_query = new_query.split("\n")[0].strip().strip('"').strip("'")
        return new_query if new_query else self.current_query


class ControllablePersonaAgent(NaiveLLMAgent):
    """Baseline: Controllable Persona Agent with Dynamic Trait Scaling (inspired by RecUserSim WWW 2025).

    Modulates its persona trait descriptions dynamically in real-time based on the perceived
    relevance and utility of the search results page (SERP).
    """

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        # Count organic result matches in SERP to estimate page utility
        relevance_score = 1.0 if "organic" in serp_description.lower() else 0.2
        # Dynamic patience and expertise descriptions scaled by SERP quality
        dynamic_patience = "EXTREMELY LOW" if relevance_score < 0.5 else self.persona.patience_level.upper()
        dynamic_style = "highly impulsive and frustrated" if relevance_score < 0.5 else self.persona.decision_style

        dynamic_desc = (
            f"You are searching for: {self.persona.search_topic}.\n"
            f"Your current dynamic state:\n"
            f"- Patience Level: {dynamic_patience} (due to SERP relevance)\n"
            f"- Search Expertise: {self.persona.search_expertise}\n"
            f"- Decision Style: {dynamic_style}"
        )

        history_lines = []
        for entry in self.memory.entries:
            history_lines.append(f"Step {entry.step}: Query '{entry.query}', Action '{entry.action}', Target '{entry.target}'")
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."

        prompt = (
            f"You are simulating a search user with dynamic traits.\n"
            f"State Profile:\n{dynamic_desc}\n\n"
            f"Search topic: {self.current_query}\n"
            f"SERP results:\n{serp_description}\n\n"
            f"Previous actions:\n{memory_context}\n\n"
            f"Allowed actions: {list(MACRO_ACTIONS)}\n"
            f"Select your next action."
        )
        action_probs = self.llm.get_action_logprobs(prompt, ACTION_LABELS)

        named_probs = {}
        for label, action in zip(ACTION_LABELS, MACRO_ACTIONS):
            named_probs[action] = action_probs.get(label, 0.0)

        # Apply basic session dynamics
        named_probs = self._apply_session_dynamics(named_probs)

        actions = list(named_probs.keys())
        probs = np.array([named_probs[a] for a in actions])
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()

        chosen_idx = np.random.choice(len(actions), p=probs)
        return actions[chosen_idx], named_probs


class SelfReflectionAgent(NaiveLLMAgent):
    """Baseline: Self-Correcting LLM User Simulator (inspired by Self-Correcting Simulators, WSDM 2026 / AAAI 2026).

    Instructs the LLM to output a explicit "Reflection" critiquing its previous steps,
    self-correcting any bad choices, and planning its next move before selecting the "Action".
    """

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        history_lines = []
        for entry in self.memory.entries:
            history_lines.append(f"Step {entry.step}: Query '{entry.query}', Action '{entry.action}', Target '{entry.target}'")
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."

        persona_desc = (
            f"You are searching for: {self.persona.search_topic}.\n"
            f"Your traits:\n"
            f"- Patience Level: {self.persona.patience_level}\n"
            f"- Search Expertise: {self.persona.search_expertise}\n"
            f"- Ad Sensitivity: {self.persona.ad_sensitivity}\n"
            f"- Decision Style: {self.persona.decision_style}"
        )

        prompt = (
            f"You are simulating a human user searching the web.\n"
            f"Persona context:\n{persona_desc}\n\n"
            f"Search topic: {self.current_query}\n"
            f"SERP results:\n{serp_description}\n\n"
            f"Previous action history:\n{memory_context}\n\n"
            f"Allowed actions: {list(MACRO_ACTIONS)}\n\n"
            f"Use the Self-Reflection format to decide your next step. You must strictly output in the following format:\n"
            f"Reflection: [Critique your previous actions, evaluate if your current search strategy is working, and self-correct any mistakes]\n"
            f"Action: [Choose exactly one action from Allowed actions]\n"
        )

        try:
            raw_response = self.llm.generate_text(prompt, max_new_tokens=150)
            action = ""
            for line in raw_response.split("\n"):
                if line.strip().startswith("Action:"):
                    action = line.replace("Action:", "").strip().lower()

            chosen_action = "skip"  # default fallback
            for a in MACRO_ACTIONS:
                if a in action or action in a:
                    chosen_action = a
                    break
        except Exception:
            chosen_action = "skip"

        probs = {a: 0.0 for a in MACRO_ACTIONS}
        probs[chosen_action] = 1.0

        return chosen_action, probs


class CoTAgent(NaiveLLMAgent):
    """Baseline: Chain-of-Thought Search Agent (inspired by NeurIPS 2024 / ICLR 2025 style CoT prompting).

    Instructs the LLM to output an explicit, detailed step-by-step reasoning chain
    before producing both action selections and query reformulations.
    """

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        history_lines = []
        for entry in self.memory.entries:
            history_lines.append(f"Step {entry.step}: Query '{entry.query}', Action '{entry.action}', Target '{entry.target}'")
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."

        persona_desc = (
            f"You are searching for: {self.persona.search_topic}.\n"
            f"Your traits:\n"
            f"- Patience Level: {self.persona.patience_level}\n"
            f"- Search Expertise: {self.persona.search_expertise}\n"
            f"- Ad Sensitivity: {self.persona.ad_sensitivity}\n"
            f"- Decision Style: {self.persona.decision_style}"
        )

        prompt = (
            f"You are simulating a human user searching the web.\n"
            f"Persona context:\n{persona_desc}\n\n"
            f"Search topic: {self.current_query}\n"
            f"SERP results:\n{serp_description}\n\n"
            f"Previous action history:\n{memory_context}\n\n"
            f"Allowed actions: {list(MACRO_ACTIONS)}\n\n"
            f"Write a step-by-step reasoning chain evaluating the current SERP page and explaining your behavioral strategy.\n"
            f"Format your response as:\n"
            f"Reasoning: [Write detailed, step-by-step reasoning explaining your choices]\n"
            f"Action: [Select exactly one action from Allowed actions]\n"
        )

        try:
            raw_response = self.llm.generate_text(prompt, max_new_tokens=150)
            action = ""
            for line in raw_response.split("\n"):
                if line.strip().startswith("Action:"):
                    action = line.replace("Action:", "").strip().lower()

            chosen_action = "skip"  # default fallback
            for a in MACRO_ACTIONS:
                if a in action or action in a:
                    chosen_action = a
                    break
        except Exception:
            chosen_action = "skip"

        probs = {a: 0.0 for a in MACRO_ACTIONS}
        probs[chosen_action] = 1.0

        return chosen_action, probs

    def generate_query(self, is_reformulation: bool = True) -> str:
        history_lines = []
        for entry in self.memory.entries:
            history_lines.append(f"Step {entry.step}: Query '{entry.query}', Action '{entry.action}', Target '{entry.target}'")
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."

        prompt = (
            f"Persona context:\n"
            f"- Topic: {self.persona.search_topic}\n"
            f"- Goal: {self.persona.search_goal}\n"
            f"Previous action history:\n{memory_context}\n\n"
            f"Formulate your next search query. You must write a step-by-step reasoning chain first.\n"
            f"Format your response strictly as:\n"
            f"Reasoning: [Step-by-step query design and term selection reasoning]\n"
            f"Query: [Write the query text ONLY]\n"
        )
        try:
            raw_response = self.llm.generate_text(prompt, max_new_tokens=150)
            new_query = self.current_query
            for line in raw_response.split("\n"):
                if line.strip().startswith("Query:"):
                    new_query = line.replace("Query:", "").strip().strip('"').strip("'")
            return new_query if new_query else self.current_query
        except Exception:
            return super().generate_query(is_reformulation)


class ValueGuidedAgent(NaiveLLMAgent):
    """Baseline: Value-Guided Decision Search Agent (inspired by Expected Utility / MCTS style simulators, WSDM 2026).

    Prompts the LLM to explicitly score the expected success / utility of each possible
    macro-action, then translates these ratings into selection probabilities via softmax.
    """

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        history_lines = []
        for entry in self.memory.entries:
            history_lines.append(f"Step {entry.step}: Query '{entry.query}', Action '{entry.action}', Target '{entry.target}'")
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."

        persona_desc = (
            f"You are searching for: {self.persona.search_topic}.\n"
            f"Your traits:\n"
            f"- Patience Level: {self.persona.patience_level}\n"
            f"- Search Expertise: {self.persona.search_expertise}"
        )

        prompt = (
            f"You are simulating a human user searching the web.\n"
            f"Persona context:\n{persona_desc}\n\n"
            f"Search topic: {self.current_query}\n"
            f"SERP results:\n{serp_description}\n\n"
            f"Previous action history:\n{memory_context}\n\n"
            f"For each candidate action below, estimate its expected utility/success score (from 1.0 to 10.0) "
            f"for advancing towards your search goal. Rate click_organic/click_result, skip, next_page, reformulate, and stop_satisfied/stop_unsatisfied/end_session/abandon.\n"
            f"Output strictly in this format (one per line, do not write anything else):\n"
            f"click_organic: [score]\n"
            f"skip: [score]\n"
            f"next_page: [score]\n"
            f"reformulate: [score]\n"
            f"abandon: [score]\n"
        )

        scores = {a: 1.0 for a in MACRO_ACTIONS}
        try:
            raw_response = self.llm.generate_text(prompt, max_new_tokens=100)
            for line in raw_response.split("\n"):
                if ":" in line:
                    parts = line.split(":")
                    act_name = parts[0].strip().lower()
                    val_str = parts[1].strip()
                    import re
                    nums = re.findall(r"\d+\.?\d*", val_str)
                    if nums:
                        val = float(nums[0])
                        # Map parsed key to MACRO_ACTIONS
                        matched_key = None
                        for a in MACRO_ACTIONS:
                            if a in act_name or act_name in a:
                                matched_key = a
                                break
                        if matched_key:
                            scores[matched_key] = np.clip(val, 1.0, 10.0)
        except Exception:
            pass

        # Softmax over ratings to convert to probabilities
        ratings = np.array([scores[a] for a in MACRO_ACTIONS])
        # scale rating to standard logits
        logits = ratings / 2.0
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()

        named_probs = {a: float(probs[i]) for i, a in enumerate(MACRO_ACTIONS)}
        chosen_idx = np.random.choice(len(MACRO_ACTIONS), p=probs)
        return MACRO_ACTIONS[chosen_idx], named_probs


# ============================================================================
# Top-venue (CCF-A) LLM user-simulation baselines, 2024-2025.
# USimAgent (SIGIR'24) is defined earlier; BASES, Agent4Rec, SimUSER,
# ConvSearch-R1 follow. Each is mechanistically distinct (not a prompt variant).
# ============================================================================


class BASESAgent(NaiveLLMAgent):
    """Baseline: BASES — large-scale web-search user simulation (Ren et al., ACL 2024).

    Reference: "BASES: Large-scale Web Search User Simulation with Large Language
    Model based Agents" (ACL 2024). Two design choices distinguish it: (1) a
    structured multi-attribute user profile is synthesized up front, and (2)
    query-behavior and click-behavior are driven by SEPARATE prompts — the paper
    found that bundling all behavior into one prompt produced "arbitrary clicks",
    so click selection is its own dedicated decision requiring an explicit page
    choice. We follow that split here.
    """

    def __init__(self, persona: Persona, llm: LLMBackbone, config: SearchAgentConfig):
        super().__init__(persona, llm, config)
        # (1) Synthesized structured profile (multi-attribute, BASES-style)
        self._profile = (
            f"User profile:\n"
            f"- Goal: {persona.search_topic}\n"
            f"- Patience: {persona.patience_level}\n"
            f"- Expertise: {persona.search_expertise}\n"
            f"- Ad sensitivity: {persona.ad_sensitivity}\n"
            f"- Decision style: {persona.decision_style}"
        )

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        # Dedicated query/navigation-behavior prompt (click is a separate decision)
        from searchagent.agent.prompts import build_action_selection_prompt
        history_lines = [
            f"Step {e.step}: Query '{e.query}', Action '{e.action}', Target '{e.target}'"
            for e in self.memory.entries
        ]
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."
        prompt = build_action_selection_prompt(
            persona_desc=self._profile,
            current_query=self.current_query,
            serp_description=serp_description,
            memory_context=memory_context,
            intent_description=f"Search goal: {self.persona.search_topic}",
        )
        action_probs = self.llm.get_action_logprobs(prompt, ACTION_LABELS)
        named_probs = {action: action_probs.get(label, 0.0)
                       for label, action in zip(ACTION_LABELS, MACRO_ACTIONS)}
        named_probs = self._apply_session_dynamics(named_probs)
        actions = list(named_probs.keys())
        probs = np.array([named_probs[a] for a in actions])
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()
        return actions[int(np.random.choice(len(actions), p=probs))], named_probs

    def select_click_target(self, serp_description: str, click_type: str,
                            serp_results: Optional[list] = None) -> int:
        """BASES dedicated click-behavior prompt: explicit page choice + justification."""
        if not serp_results:
            return 0
        candidates = serp_results
        if click_type == "click_ad":
            idxs = [i for i, r in enumerate(serp_results) if r.is_ad]
        elif click_type == "click_organic":
            idxs = [i for i, r in enumerate(serp_results) if not r.is_ad]
        else:
            idxs = list(range(len(serp_results)))
        if not idxs:
            return 0
        # Ask the LLM which result this profiled user would click (forced choice).
        listing = "\n".join(
            f"{chr(65+j)}. {serp_results[i].title}" + (" [Ad]" if serp_results[i].is_ad else "")
            for j, i in enumerate(idxs[:7])
        )
        labels = [chr(65 + j) for j in range(min(len(idxs), 7))]
        prompt = (
            f"{self._profile}\n\nQuery: {self.current_query}\n"
            f"Which ONE result does this user click? Reply a single letter.\n{listing}"
        )
        try:
            probs = self.llm.get_action_logprobs(prompt, labels)
            best = max(labels, key=lambda l: probs.get(l, -1e9))
            return idxs[labels.index(best)]
        except Exception:
            return idxs[0]

    def generate_query(self, is_reformulation: bool = True) -> str:
        from searchagent.agent.prompts import build_query_generation_prompt
        history_lines = [
            f"Step {e.step}: Query '{e.query}', Action '{e.action}'" for e in self.memory.entries
        ]
        memory_context = "\n".join(history_lines) if history_lines else "No previous actions."
        prompt = build_query_generation_prompt(
            persona_desc=self._profile, original_query=self.current_query,
            memory_context=memory_context,
            intent_description=f"Search goal: {self.persona.search_topic}",
            is_reformulation=is_reformulation,
        )
        q = self.llm.generate_text(prompt, max_new_tokens=32).split("\n")[0].strip().strip('"').strip("'")
        return q if q else self.current_query







class Agent4RecAgent(NaiveLLMAgent):
    """Baseline: Agent4Rec — generative recommender agent (Zhang et al., SIGIR 2024).

    Reference: "On Generative Agents in Recommendation" (SIGIR 2024). Models a
    user via three modules: a taste/profile module, a memory module, and an
    action module. The agent forms a free-text "taste" summary, retrieves
    relevant memory, and emits taste-driven engagement actions (click / skip /
    next / exit). Distinct from plain prompting via the explicit taste+memory
    conditioning and an emotion/satisfaction-gated stopping tendency.
    """

    def __init__(self, persona: Persona, llm: LLMBackbone, config: SearchAgentConfig):
        super().__init__(persona, llm, config)
        self._taste = (
            f"Taste: interested in '{persona.search_topic}'. "
            f"Expertise {persona.search_expertise}, ad-sensitivity {persona.ad_sensitivity}, "
            f"decision style {persona.decision_style}."
        )

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        from searchagent.agent.prompts import build_action_selection_prompt
        # Memory module: retrieve recent engagement
        recent = self.memory.entries[-3:]
        mem = "; ".join(f"{e.action}@step{e.step}" for e in recent) if recent else "none"
        persona_desc = (
            f"{self._taste}\nRecent engagement memory: {mem}\n"
            f"Satisfaction so far: {self._last_satisfaction:.1f}/5."
        )
        prompt = build_action_selection_prompt(
            persona_desc=persona_desc,
            current_query=self.current_query,
            serp_description=serp_description,
            memory_context=mem,
            intent_description=f"Taste-driven browsing of: {self.persona.search_topic}",
        )
        action_probs = self.llm.get_action_logprobs(prompt, ACTION_LABELS)
        named_probs = {action: action_probs.get(label, 0.0)
                       for label, action in zip(ACTION_LABELS, MACRO_ACTIONS)}
        # Emotion/satisfaction-gated exit (Agent4Rec): satisfied users leave sooner
        if self._last_satisfaction >= 4.0:
            for k in ("stop_satisfied",):
                if k in named_probs:
                    named_probs[k] *= 2.5
        named_probs = self._apply_session_dynamics(named_probs)
        actions = list(named_probs.keys())
        probs = np.array([named_probs[a] for a in actions])
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()
        return actions[int(np.random.choice(len(actions), p=probs))], named_probs

class SimUSERAgent(NaiveLLMAgent):
    """Baseline: SimUSER — persona-inference user simulator (Bernard et al., ACL 2025).

    Reference: "Simulating User Behavior with Large Language Models for Recommender
    System Evaluation" (ACL 2025, Industry). Phase 1 INFERS a self-consistent
    persona from the user's interaction history rather than using a handed-in
    profile; Phase 2 runs a persona+memory+perception+brain agent. We implement
    the two phases: an inferred-persona string (refreshed from accumulated history)
    drives the action decision. Distinct via the inferred (not given) persona.
    """

    def __init__(self, persona: Persona, llm: LLMBackbone, config: SearchAgentConfig):
        super().__init__(persona, llm, config)
        self._seed_topic = persona.search_topic
        self._inferred_persona = f"A user searching for '{persona.search_topic}'."

    def _infer_persona(self) -> None:
        """Phase 1: infer a self-consistent persona from accumulated history."""
        if not self.memory.entries:
            return
        hist = "; ".join(
            f"{e.action} '{e.target}'" for e in self.memory.entries[-5:] if e.target
        )
        if not hist:
            return
        try:
            prompt = (
                f"A user is searching for '{self._seed_topic}'. Their actions so far: {hist}.\n"
                f"In one sentence, describe this user's persona (intent, patience, preferences)."
            )
            inferred = self.llm.generate_text(prompt, max_new_tokens=60).strip()
            if inferred:
                self._inferred_persona = inferred
        except Exception:
            pass

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        from searchagent.agent.prompts import build_action_selection_prompt
        self._infer_persona()  # Phase 1: refresh inferred persona
        recent = self.memory.entries[-3:]
        mem = "; ".join(f"{e.action}@{e.step}" for e in recent) if recent else "none"
        prompt = build_action_selection_prompt(
            persona_desc=f"Inferred persona: {self._inferred_persona}",
            current_query=self.current_query,
            serp_description=serp_description,
            memory_context=mem,
            intent_description=f"Pursuing: {self._seed_topic}",
        )
        action_probs = self.llm.get_action_logprobs(prompt, ACTION_LABELS)
        named_probs = {action: action_probs.get(label, 0.0)
                       for label, action in zip(ACTION_LABELS, MACRO_ACTIONS)}
        named_probs = self._apply_session_dynamics(named_probs)
        actions = list(named_probs.keys())
        probs = np.array([named_probs[a] for a in actions])
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()
        return actions[int(np.random.choice(len(actions), p=probs))], named_probs

    def reset(self) -> None:
        super().reset()
        self._inferred_persona = f"A user searching for '{self._seed_topic}'."


class ConvSearchR1Agent(NaiveLLMAgent):
    """Baseline: ConvSearch-R1 — reasoning-driven query reformulation (Zhu et al., EMNLP 2025).

    Reference: "Enhancing Query Reformulation for Conversational Search with
    Reasoning via Reinforcement Learning" (EMNLP 2025). Its contribution is on
    the REFORMULATION sub-task: the model first produces an explicit reasoning
    chain over the search context, then emits a context-resolved rewritten query
    (the RL reward in the paper is retrieval-grounded; we keep the reason-then-
    rewrite inference behavior). Action choice uses the base LLM; the distinctive
    component is the reasoning-grounded query rewrite.
    """

    def generate_query(self, is_reformulation: bool = True) -> str:
        history_lines = [
            f"Step {e.step}: '{e.query}' -> {e.action}" for e in self.memory.entries
        ]
        context = "\n".join(history_lines) if history_lines else "(no prior turns)"
        try:
            prompt = (
                f"You reformulate a web-search query using reasoning.\n"
                f"Original goal: {self.persona.search_topic}\n"
                f"Current query: {self.current_query}\n"
                f"Search history:\n{context}\n\n"
                f"First reason about what is missing or ambiguous, then output the "
                f"improved standalone query on the FINAL line prefixed exactly with 'QUERY: '."
            )
            out = self.llm.generate_text(prompt, max_new_tokens=120)
            q = ""
            for line in out.splitlines():
                if "QUERY:" in line:
                    q = line.split("QUERY:", 1)[1].strip().strip('"').strip("'")
            if not q:
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                q = lines[-1].strip('"').strip("'") if lines else ""
            return q if q else self.current_query
        except Exception:
            return super().generate_query(is_reformulation)

