"""Core search user agent that orchestrates LLM-driven decision making."""

from __future__ import annotations
import logging
import random
from typing import Any, Dict, List, Optional, Tuple
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
    build_satisfaction_prompt,
)
from searchagent.config import IntentConfig, SearchAgentConfig

logger = logging.getLogger(__name__)






class CognitiveSearchState:
    """Cognitive Search State (CSS) module.

    Models the evolving cognitive process during information search,
    grounded in two foundational IR theories:

    1. Kuhlthau's Information Search Process (ISP, 1991):
       Users progress through affective stages (uncertainty → optimism
       → confusion/frustration → clarity → satisfaction) that directly
       influence search behavior.

    2. Belkin's Anomalous States of Knowledge (ASK, 1980):
       Information needs arise from gaps in knowledge; the search
       process is driven by resolving these anomalies.

    Four cognitive dimensions:
      - need_clarity:      How clear the information need is (0→1)
      - knowledge_state:   Fraction of need fulfilled (0→1)
      - frustration:       Accumulated frustration (0→1)
      - exploration_focus:  Exploration (0) vs Exploitation (1)

    Unlike Agent4Rec's binary emotion labels (happy/bored), CSS
    provides continuous, multi-dimensional cognitive tracking that
    feeds into BOTH the LLM prompt context AND probability modulation.
    """

    def __init__(self):
        self.need_clarity = 0.3      # Initial ambiguity (ASK)
        self.knowledge_state = 0.0   # No knowledge yet
        self.frustration = 0.0       # Fresh start
        self.exploration_focus = 0.3  # Slightly exploratory
        self._interaction_count = 0

    def update(self, action: str, satisfaction: float = 0.0) -> None:
        """Update cognitive state based on action and outcome.

        Transitions follow Kuhlthau's ISP phases:
        - Satisfying click → clarity + knowledge gain (Collection → Presentation)
        - Unsatisfying click → frustration (Exploration → Confusion)
        - Reformulation → need clarification attempt (Formulation)
        - Skip → mild frustration accumulation
        """
        self._interaction_count += 1

        if action in ("click_organic", "click_ad", "click_result"):
            if satisfaction >= 4.0:
                # ISP "Collection" → high clarity, knowledge gained
                self.knowledge_state = min(1.0, self.knowledge_state + 0.15)
                self.need_clarity = min(1.0, self.need_clarity + 0.1)
                self.frustration = max(0.0, self.frustration - 0.1)
                self.exploration_focus = min(1.0, self.exploration_focus + 0.1)
            elif satisfaction >= 3.0:
                # Moderate progress
                self.knowledge_state = min(1.0, self.knowledge_state + 0.05)
                self.need_clarity = min(1.0, self.need_clarity + 0.05)
                self.exploration_focus = min(1.0, self.exploration_focus + 0.05)
            else:
                # ISP "Confusion" → frustration, learn what you DON'T want
                self.frustration = min(1.0, self.frustration + 0.1)
                self.need_clarity = min(1.0, self.need_clarity + 0.03)

        elif action == "reformulate":
            # ISP "Formulation" → active need clarification
            self.need_clarity = min(1.0, self.need_clarity + 0.1)
            self.exploration_focus = max(0.0, self.exploration_focus - 0.2)
            self.frustration = min(1.0, self.frustration + 0.1)

        elif action == "skip":
            self.frustration = min(1.0, self.frustration + 0.1)

        elif action == "next_page":
            self.exploration_focus = max(0.0, self.exploration_focus - 0.1)
            self.frustration = min(1.0, self.frustration + 0.1)

    def get_action_modifiers(self) -> Dict[str, float]:
        """Probability multipliers derived from cognitive state.

        These are GENTLE nudges (not strong overrides) because CSS's
        primary contribution is prompt context for the LLM. The existing
        session dynamics (satiation, fatigue, satisfaction feedback)
        already handle strong behavioral constraints.
        """
        m = {a: 1.0 for a in [
            "click_result", "click_organic", "click_ad", "skip", "next_page",
            "reformulate", "new_query", "abandon", "end_session",
            "stop_satisfied", "stop_unsatisfied",
        ]}

        # High knowledge → gentle nudge toward ending
        if self.knowledge_state > 0.6:
            for k in ("stop_satisfied", "end_session"):
                m[k] *= (1.0 + self.knowledge_state * 0.5)  # up to 1.5x

        # High frustration → strong push to reformulate and abandon
        if self.frustration > 0.3:
            m["reformulate"] *= (1.0 + self.frustration * 1.5)  # up to 2.5x
            m["new_query"] *= (1.0 + self.frustration * 1.0)
            m["stop_unsatisfied"] *= (1.0 + self.frustration * 2.0)  # up to 3.0x push to abandon!

        # High exploitation focus → slight click boost
        if self.exploration_focus > 0.6:
            for k in ("click_result", "click_organic"):
                m[k] *= (1.0 + self.exploration_focus * 0.2)

        return m

    def describe(self) -> str:
        """Natural language description for LLM prompt context.

        Translates numeric cognitive state into human-readable text
        that the LLM can reason about.
        """
        clarity_map = [
            (0.3, "very unclear"), (0.5, "somewhat unclear"),
            (0.7, "fairly clear"), (1.1, "very clear"),
        ]
        knowledge_map = [
            (0.2, "nothing useful yet"), (0.5, "some relevant information"),
            (0.8, "substantial information"), (1.1, "nearly everything needed"),
        ]
        focus_map = [
            (0.3, "exploring broadly"), (0.6, "narrowing your focus"),
            (1.1, "focused on a specific lead"),
        ]

        def _lookup(val, table):
            for threshold, desc in table:
                if val < threshold:
                    return desc
            return table[-1][1]

        parts = [
            f"Your information need is {_lookup(self.need_clarity, clarity_map)}.",
            f"You have found {_lookup(self.knowledge_state, knowledge_map)}.",
            f"You are {_lookup(self.exploration_focus, focus_map)}.",
        ]
        if self.frustration >= 0.2:
            if self.frustration < 0.4:
                parts.append("You are getting slightly frustrated.")
            elif self.frustration < 0.7:
                parts.append("You are noticeably frustrated with the search.")
            else:
                parts.append("You are very frustrated and losing patience.")
        return " ".join(parts)

    def reset(self) -> None:
        self.need_clarity = 0.3
        self.knowledge_state = 0.0
        self.frustration = 0.0
        self.exploration_focus = 0.3
        self._interaction_count = 0







class IntentTracker:
    """Tracks soft intent distribution over session steps."""

    def __init__(self, config: IntentConfig):
        self.states = config.states
        self.transitions = config.transitions
        self.distribution = np.array(config.initial_distribution, dtype=np.float64)
        self.distribution /= self.distribution.sum()

    def update(self, action: str, momentum: float = 0.3) -> None:
        """Update intent distribution based on the action taken."""
        if action in self.transitions and self.transitions[action]:
            target = np.array(self.transitions[action], dtype=np.float64)
            if target.sum() > 0:
                target /= target.sum()
                self.distribution = momentum * self.distribution + (1 - momentum) * target
                self.distribution /= self.distribution.sum()

    def describe(self) -> str:
        """Human-readable intent description for prompts."""
        parts = []
        for state, prob in zip(self.states, self.distribution):
            if prob > 0.1:
                parts.append(f"{state} ({prob:.0%})")
        return "Intent: " + ", ".join(parts)

    def dominant_intent(self) -> str:
        return self.states[int(np.argmax(self.distribution))]

    def reset(self, initial: Optional[List[float]] = None) -> None:
        if initial is not None:
            self.distribution = np.array(initial, dtype=np.float64)
        self.distribution /= self.distribution.sum()







class SearchUserAgent:
    """LLM-powered search user agent.

    Orchestrates persona, memory, intent tracking, and LLM calls
    to simulate realistic search behavior including ad interactions.
    """

    def __init__(
        self,
        persona: Persona,
        llm: LLMBackbone,
        config: SearchAgentConfig,
        calibrator=None,
        dataset_stats=None,
    ):
        self.persona = persona
        self.llm = llm
        self.config = config
        self.memory = AgentMemory()
        self.intent = IntentTracker(config.intent)
        self.cognitive_state = CognitiveSearchState()
        self.current_query = persona.search_topic
        self.step_count = 0
        self._persona_desc = persona.to_prompt_description()
        self._last_satisfaction = 0.0  # tracks satisfaction of last click
        self._total_clicks = 0
        self._total_skips = 0
        self._ad_clicked = 0
        # SearchAgent++ : learned posterior calibrator. When set, decide_action and
        # select_click_target use the trained calibration policy instead of the
        # hand-weighted MNL utilities. None -> original v1 path (kept for ablations).
        self.calibrator = calibrator
        # Dataset-aware stats [clicks/session_norm, abandon, reform] for the
        # calibrator's per-dataset adaptation. Defaults to neutral zeros.
        self._dataset_stats = dataset_stats if dataset_stats is not None else [0.0, 0.0, 0.0]
        if calibrator is not None:
            from searchagent.agent.calibration import StateFeaturizer
            self._featurizer = StateFeaturizer(n_intent=len(config.intent.states))

    def decide_action(self, serp_description: str) -> Tuple[str, Dict[str, float]]:
        """Decide the next macro-action via the learned posterior calibrator.

        P_theta(a | Omega_t)  proportional to  p_LLM(a)^{alpha_theta}  *  exp(f_theta(a, Omega_t))
        """
        prompt = build_action_selection_prompt(
            persona_desc=self._persona_desc,
            current_query=self.current_query,
            serp_description=serp_description,
            memory_context=self.memory.get_recent_context(),
            intent_description=self.intent.describe(),
            css_context=self.cognitive_state.describe(),
        )
        action_probs = self.llm.get_action_logprobs(prompt, ACTION_LABELS)
        return self._decide_action_calibrated(serp_description, action_probs)

    def _decide_action_calibrated(self, serp_description, action_probs):
        """SearchAgent++ action decision via learned posterior P_theta(a|Omega)."""
        from searchagent.agent.calibration import CALIB_ACTIONS
        # LLM prior in MACRO_ACTIONS order -> CALIB_ACTIONS order.
        prior = np.array([action_probs.get(l, 1e-8) for l in ACTION_LABELS], dtype=np.float32)
        prior_macro = dict(zip(MACRO_ACTIONS, prior))
        a_prior = np.array([prior_macro.get(a, 1e-8) for a in CALIB_ACTIONS], dtype=np.float32)
        a_prior = a_prior / a_prior.sum()
        log_p = np.log(np.clip(a_prior, 1e-9, 1.0)).astype(np.float32)

        # SERP quality proxy
        sq = self._serp_quality(serp_description)
        mem_feats = dict(
            n_click=min(self._total_clicks / 5.0, 1.0),
            n_skip=min(self._total_skips / 5.0, 1.0),
            ad_clicked=float(self._ad_clicked > 0),
            last_satisfaction=self._last_satisfaction / 5.0,
            page=1.0, step=min(self.step_count / 10.0, 1.0),
        )
        sf = self._featurizer.build(
            self.persona, self.cognitive_state, list(self.intent.distribution),
            mem_feats, sq, a_prior, dataset_stats=self._dataset_stats)
        gate = np.array([sf.llm_entropy, min(self.step_count / 10.0, 1.0),
                         min(self._total_clicks / 5.0, 1.0),
                         self.cognitive_state.frustration,
                         self.cognitive_state.knowledge_state, sq], dtype=np.float32)

        probs = self.calibrator.action_probs(sf.vector(), log_p, gate)
        probs = np.clip(probs, 1e-9, None); probs /= probs.sum()
        # Click soft-cap: the calibrator occasionally keeps emitting high click
        # probability on unlucky seeds, blowing up clicks/session (e.g. 9 vs real
        # 4.4). When total clicks exceed the per-dataset target, decay click_result
        # probability (mass shifts to stop/reformulate) so the count stays calibrated.
        target = getattr(self.config.calibration, "target_clicks", 0.0)
        if target and "click_result" in CALIB_ACTIONS:
            ci = CALIB_ACTIONS.index("click_result")
            if self._total_clicks >= target:
                excess = self._total_clicks - target + 1.0
                probs[ci] *= 0.5 ** excess   # halve per click over target
            else:
                # Below target: boost click prob for under-clicking datasets (yandex).
                boost = getattr(self.config.calibration, "click_boost", 1.0)
                if boost and boost != 1.0:
                    probs[ci] *= boost
            probs = np.clip(probs, 1e-9, None); probs /= probs.sum()
        named = dict(zip(CALIB_ACTIONS, probs))
        # Map CALIB action back to a MACRO action the simulator understands.
        idx = int(np.random.choice(len(CALIB_ACTIONS), p=probs))
        chosen = CALIB_ACTIONS[idx]
        return chosen, named

    def _serp_quality(self, serp_description) -> float:
        """Cheap SERP-quality proxy in [0,1] from description length/diversity."""
        if not serp_description:
            return 0.5
        lines = [l for l in serp_description.splitlines() if l.strip()]
        return float(min(len(lines) / 20.0, 1.0))



    def select_click_target(
        self, serp_description: str, click_type: str,
        serp_results: Optional[list] = None,
    ) -> int:
        """Select which specific result to click via the learned ClickModel."""
        if serp_results is None:
            serp_results = []

        if click_type == "click_ad":
            candidates = [(i, r) for i, r in enumerate(serp_results) if r.is_ad]
        else:
            candidates = list(enumerate(serp_results))

        if not candidates:
            candidates = list(enumerate(serp_results)) if serp_results else [(0, None)]

        if len(candidates) == 1:
            return candidates[0][0]

        sel = self._select_click_calibrated(candidates)
        return sel if sel is not None else candidates[0][0]

    def _select_click_calibrated(self, candidates):
        """SearchAgent++ click selection via learned examination * attractiveness.

        Builds per-result exam/attr features matching training (data_builder
        ._exam_attr_feats), scores them with the ClickModel, samples a result.
        Returns the original result index, or None to fall back to v1.
        """
        results = [r for _, r in candidates if r is not None]
        if len(results) < 2:
            return None
        try:
            import numpy as _np
            expertise_idx = (["novice", "intermediate", "expert"].index(
                getattr(self.persona, "search_expertise", "intermediate")) / 2.0)
            focus = self.cognitive_state.exploration_focus
            n = len(results)
            exam = _np.zeros((n, 7), dtype=_np.float32)
            attr = _np.zeros((n, 7), dtype=_np.float32)
            for i, r in enumerate(results):
                rank = getattr(r, "rank", i + 1)
                is_ad = 1.0 if getattr(r, "is_ad", False) else 0.0
                rel = getattr(r, "relevance", 0.5)
                exam[i] = [rank / 10.0, _np.log(rank + 1) / 3.0,
                           min(self.step_count / 10.0, 1.0), focus,
                           expertise_idx, is_ad, 1.0]
                # no per-result LLM logprob at inference -> use relevance as proxy
                attr[i] = [_np.log(max(rel, 1e-3)) / 9.0, rel, is_ad,
                           rel, rank / 10.0, expertise_idx, 1.0]
            probs = self.calibrator.click_probs(exam, attr)
            probs = _np.clip(probs, 1e-9, None); probs /= probs.sum()
            # Optional click-trace recording for offline temperature tuning.
            # Records the PRE-temperature raw probs + each candidate's rank, so an
            # offline tool can replay any temperature and compute click_position_jsd
            # with ZERO API calls (temperature only re-weights position, no LLM).
            rec = getattr(SearchUserAgent, "_CLICK_TRACE", None)
            if rec is not None:
                ranks = [int(getattr(r, "rank", i + 1)) for i, r in enumerate(results)]
                rec.append({"probs": probs.tolist(), "ranks": ranks})
            # Temperature softening: the ClickModel concentrates ~95% on rank 1,
            # but real click-position dists decay gradually (rank1~0.62, rank2~0.23).
            # T>1 flattens the dist toward the real position-bias curve.
            temp = getattr(self.config.calibration, "click_temperature", 1.0)
            if temp and temp != 1.0:
                probs = probs ** (1.0 / temp)
                probs /= probs.sum()
            local = int(_np.random.choice(n, p=probs))
            # map back to candidates index (results filtered None)
            valid = [idx for idx, (_, r) in enumerate(candidates) if r is not None]
            return candidates[valid[local]][0]
        except Exception:
            return None

    def rate_satisfaction(self, clicked_title: str, clicked_snippet: str, is_ad: bool) -> float:
        """Rate satisfaction after clicking a result (1-5 scale).

        Uses LLM logprobs over rating labels for robust extraction.
        Returns float satisfaction score.
        """
        prompt = build_satisfaction_prompt(
            persona_desc=self._persona_desc,
            clicked_title=clicked_title,
            clicked_snippet=clicked_snippet,
            is_ad=is_ad,
        )
        rating_labels = ["1", "2", "3", "4", "5"]
        try:
            probs = self.llm.get_action_logprobs(prompt, rating_labels)
            # Compute expected satisfaction as weighted average
            expected = sum(int(k) * v for k, v in probs.items())
            return max(1.0, min(5.0, expected))
        except Exception:
            return 3.0  # neutral fallback

    @staticmethod
    def estimate_dwell_time(satisfaction: float, is_ad: bool) -> float:
        """Estimate dwell time in seconds from satisfaction rating.

        Based on empirical observations:
        - Higher satisfaction → longer dwell time
        - Ad results tend to have shorter dwell times
        - Base dwell follows log-normal-ish distribution
        """
        # Base dwell time increases with satisfaction (seconds)
        base_dwell = {1: 5.0, 2: 12.0, 3: 25.0, 4: 45.0, 5: 75.0}
        base = base_dwell.get(round(satisfaction), 25.0)
        # Interpolate for non-integer satisfaction
        frac = satisfaction - int(satisfaction)
        if frac > 0 and round(satisfaction) < 5:
            next_base = base_dwell.get(round(satisfaction) + 1, base)
            base = base + frac * (next_base - base)
        # Ad discount: users spend ~40% less time on ad pages
        if is_ad:
            base *= 0.6
        # Add noise
        noise = np.random.lognormal(0, 0.3)
        return round(base * noise, 1)

    def generate_query(self, is_reformulation: bool = True) -> str:
        """Generate a new or reformulated search query."""
        prompt = build_query_generation_prompt(
            persona_desc=self._persona_desc,
            original_query=self.current_query,
            memory_context=self.memory.get_recent_context(),
            intent_description=self.intent.describe(),
            is_reformulation=is_reformulation,
        )
        new_query = self.llm.generate_text(prompt, max_new_tokens=32)
        # Clean up: take first line, remove quotes
        new_query = new_query.split("\n")[0].strip().strip('"').strip("'")
        if not new_query:
            new_query = self.current_query
        return new_query

    def record_action(
        self,
        action: str,
        target: str = "",
        result_title: str = "",
        is_ad: bool = False,
        reasoning: str = "",
        satisfaction: float = 0.0,
    ) -> None:
        """Record an action in memory and update intent."""
        entry = MemoryEntry(
            step=self.step_count,
            query=self.current_query,
            action=action,
            target=target,
            result_title=result_title,
            is_ad=is_ad,
            satisfaction=satisfaction,
            reasoning=reasoning,
        )
        self.memory.add(entry)
        self.intent.update(action)
        self.cognitive_state.update(action, satisfaction)
        if action in ("click_organic", "click_ad", "click_result"):
            self._total_clicks += 1
            self._last_satisfaction = satisfaction
            if action == "click_ad" or is_ad:
                self._ad_clicked += 1
        elif action == "skip":
            self._total_skips += 1
        self.step_count += 1

    def reset(self) -> None:
        """Reset agent state for a new session."""
        self.memory.reset()
        self.intent.reset(self.config.intent.initial_distribution)
        self.cognitive_state.reset()
        self.current_query = self.persona.search_topic
        self.step_count = 0
        self._last_satisfaction = 0.0
        self._total_clicks = 0
        self._total_skips = 0
        self._ad_clicked = 0
