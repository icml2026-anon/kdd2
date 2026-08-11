from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from searchagent.agent.memory import AgentMemory, MemoryEntry
from searchagent.agent.persona import Persona
from searchagent.agent.prompts import ACTION_LABELS, MACRO_ACTIONS
from searchagent.config import SearchAgentConfig


_PHI = dict(zip(ACTION_LABELS, MACRO_ACTIONS))


def _canonical_query(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\s\.,;:!?]+$", "", value)
    return value


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        current = [i]
        for j, ca in enumerate(a, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalized_levenshtein_distance(a: str, b: str) -> float:
    ca = _canonical_query(a)
    cb = _canonical_query(b)
    denom = max(len(ca), len(cb), 1)
    return _levenshtein(ca, cb) / denom


def retained_token_fraction(current_query: str, generated_query: str) -> float:
    current = _canonical_query(current_query).split()
    generated = _canonical_query(generated_query).split()
    if not current:
        return 0.0
    remaining: Dict[str, int] = {}
    for token in generated:
        remaining[token] = remaining.get(token, 0) + 1
    kept = 0
    for token in current:
        if remaining.get(token, 0) > 0:
            kept += 1
            remaining[token] -= 1
    return kept / len(current)


class BaselineAdapter:
    @staticmethod
    def cot(label: str) -> Optional[str]:
        return _PHI.get(str(label).strip().upper())

    @staticmethod
    def usersimcrs(act: str, has_item: bool = False, satisfaction: Optional[float] = None) -> Optional[str]:
        name = str(act).strip().lower().replace("-", "_")
        if name in {"accept", "inspect"}:
            return "click_result" if has_item else None
        if name in {"reject", "continue"}:
            return "skip"
        if name in {"request", "request_more", "requestmore"}:
            return "next_page"
        if name == "clarify":
            return "reformulate"
        if name == "end" and satisfaction is not None:
            return "stop_satisfied" if float(satisfaction) >= 0.5 else "stop_unsatisfied"
        return None

    @staticmethod
    def cshi(act: str, satisfaction: Optional[float] = None) -> Optional[str]:
        name = str(act).strip().lower().replace("-", "_")
        direct = {
            "click": "click_result",
            "continue": "skip",
            "refine": "reformulate",
            "switch": "new_query",
        }
        if name in direct:
            return direct[name]
        if name == "finish" and satisfaction is not None:
            return "stop_satisfied" if float(satisfaction) >= 0.5 else "stop_unsatisfied"
        return None

    @staticmethod
    def usimagent(event: str, current_query: str = "", generated_query: str = "", browse_mode: str = "", stop_state: str = "") -> Optional[str]:
        name = str(event).strip().lower().replace("-", "_")
        if name == "click":
            return "click_result"
        if name == "browse":
            mode = str(browse_mode).strip().lower().replace("-", "_")
            if mode in {"next", "next_page", "page"}:
                return "next_page"
            if mode in {"skip", "continue", "same_page"}:
                return "skip"
            return None
        if name == "query" and generated_query:
            return "reformulate" if normalized_levenshtein_distance(current_query, generated_query) < 0.5 else "new_query"
        if name == "stop":
            state = str(stop_state).strip().lower()
            if state in {"satisfied", "success", "successful"}:
                return "stop_satisfied"
            if state in {"unsatisfied", "failure", "failed"}:
                return "stop_unsatisfied"
        return None

    @staticmethod
    def bases(event: str, current_query: str = "", generated_query: str = "", browse_mode: str = "", terminal_utility: Optional[float] = None) -> Optional[str]:
        name = str(event).strip().lower().replace("-", "_")
        if name == "click":
            return "click_result"
        if name == "browse":
            mode = str(browse_mode).strip().lower().replace("-", "_")
            if mode in {"next", "next_page", "page"}:
                return "next_page"
            if mode in {"skip", "continue", "same_page"}:
                return "skip"
            return None
        if name == "query" and generated_query:
            return "reformulate" if normalized_levenshtein_distance(current_query, generated_query) < 0.5 else "new_query"
        if name in {"stop", "finish", "end"} and terminal_utility is not None:
            return "stop_satisfied" if float(terminal_utility) >= 3.0 else "stop_unsatisfied"
        return None

    @staticmethod
    def agent4rec(event: str, page_boundary: Optional[bool] = None, exit_rating: str = "") -> Optional[str]:
        name = str(event).strip().lower().replace("-", "_")
        if name in {"item_selection", "item_select", "select", "click"}:
            return "click_result"
        if name in {"browse", "browsing"} and page_boundary is not None:
            return "next_page" if bool(page_boundary) else "skip"
        if name in {"preference_refinement", "refine_preference", "refine"}:
            return "reformulate"
        if name in {"goal_shift", "switch_goal", "shift_goal"}:
            return "new_query"
        if name in {"exit", "finish", "stop"}:
            rating = str(exit_rating).strip().lower()
            if rating in {"satisfied", "positive", "success", "successful"}:
                return "stop_satisfied"
            if rating in {"unsatisfied", "negative", "failure", "failed"}:
                return "stop_unsatisfied"
        return None

    @staticmethod
    def simuser(act: str, finish_state: str = "") -> Optional[str]:
        name = str(act).strip().lower().replace("-", "_")
        direct = {
            "item_choice": "click_result",
            "choose_item": "click_result",
            "continue": "skip",
            "request_more": "next_page",
            "requestmore": "next_page",
            "refine": "reformulate",
            "switch_goal": "new_query",
        }
        if name in direct:
            return direct[name]
        if name == "finish":
            state = str(finish_state).strip().lower()
            if state in {"satisfied", "success", "successful"}:
                return "stop_satisfied"
            if state in {"unsatisfied", "failure", "failed"}:
                return "stop_unsatisfied"
        return None

    @staticmethod
    def convsearch(current_query: str, generated_query: str) -> Optional[str]:
        if not str(generated_query).strip():
            return None
        return "reformulate" if retained_token_fraction(current_query, generated_query) >= 0.5 else "new_query"


@dataclass
class _NativeDecision:
    action: str
    click_rank: Optional[int] = None
    query: str = ""


class _BaseAgent:
    def __init__(self, persona: Persona, config: SearchAgentConfig):
        self.persona = persona
        self.config = config
        self.current_query = persona.search_topic
        self.step_count = 0
        self.memory = AgentMemory()
        self._total_clicks = 0
        self._last_satisfaction = 0.0
        self._pending_click_rank: Optional[int] = None
        self._pending_query = ""

    def _one_hot(self, action: str) -> Dict[str, float]:
        return {name: float(name == action) for name in MACRO_ACTIONS}

    def _set_decision(self, decision: _NativeDecision) -> Tuple[str, Dict[str, float]]:
        self._pending_click_rank = decision.click_rank
        self._pending_query = decision.query.strip()
        return decision.action, self._one_hot(decision.action)

    def select_click_target(self, serp_description: Any, click_type: str, serp_results: Optional[list] = None) -> int:
        results = list(serp_results or [])
        if not results:
            return 0
        if self._pending_click_rank is not None:
            rank = self._pending_click_rank
            self._pending_click_rank = None
            for index, result in enumerate(results):
                if int(getattr(result, "rank", index + 1)) == rank:
                    return index
        return 0

    def generate_query(self, is_reformulation: bool = True) -> str:
        if self._pending_query:
            query = self._pending_query
            self._pending_query = ""
            return query
        return self.current_query

    def record_action(self, action: str, target: str = "", result_title: str = "", is_ad: bool = False, reasoning: str = "", satisfaction: float = 0.0) -> None:
        self.memory.add(MemoryEntry(step=self.step_count, query=self.current_query, action=action, target=target, result_title=result_title, is_ad=is_ad, reasoning=reasoning))
        if action == "click_result":
            self._total_clicks += 1
            self._last_satisfaction = float(satisfaction or 0.0)
        self.step_count += 1

    def reset(self) -> None:
        self.current_query = self.persona.search_topic
        self.step_count = 0
        self.memory.reset()
        self._total_clicks = 0
        self._last_satisfaction = 0.0
        self._pending_click_rank = None
        self._pending_query = ""


class RandomAgent(_BaseAgent):
    def __init__(self, persona: Persona, config: SearchAgentConfig):
        super().__init__(persona, config)
        self._action_probs = np.full(len(MACRO_ACTIONS), 1.0 / len(MACRO_ACTIONS), dtype=np.float64)

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        action = str(np.random.choice(MACRO_ACTIONS, p=self._action_probs))
        return action, dict(zip(MACRO_ACTIONS, self._action_probs.tolist()))

    def select_click_target(self, serp_description: Any, click_type: str, serp_results: Optional[list] = None) -> int:
        results = list(serp_results or [])
        if not results:
            return 0
        if click_type == "click_ad":
            candidates = [i for i, r in enumerate(results) if bool(getattr(r, "is_ad", False))]
        elif click_type == "click_organic":
            candidates = [i for i, r in enumerate(results) if not bool(getattr(r, "is_ad", False))]
        else:
            candidates = list(range(len(results)))
        if not candidates:
            candidates = list(range(len(results)))
        return int(np.random.choice(candidates))

    def generate_query(self, is_reformulation: bool = True) -> str:
        suffix = str(np.random.choice(["review", "comparison", "best", "guide"]))
        return f"{self.current_query} {suffix}".strip()


class RuleBasedAgent(_BaseAgent):
    def __init__(self, persona: Persona, config: SearchAgentConfig):
        super().__init__(persona, config)
        self.pos_bias = np.asarray(config.environment.position_bias_click_prob, dtype=np.float64)
        self._pages_without_click = 0
        self._max_pages_before_abandon = 3

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        if self._pages_without_click >= self._max_pages_before_abandon:
            return "stop_unsatisfied", self._one_hot("stop_unsatisfied")
        action = "click_result" if np.random.random() < float(self.pos_bias[0]) else "skip"
        return action, self._one_hot(action)

    def select_click_target(self, serp_description: Any, click_type: str, serp_results: Optional[list] = None) -> int:
        results = list(serp_results or [])
        if not results:
            return 0
        weights = np.asarray([self.pos_bias[min(max(int(getattr(r, "rank", i + 1)) - 1, 0), len(self.pos_bias) - 1)] for i, r in enumerate(results)], dtype=np.float64)
        weights = np.clip(weights, 1e-12, None)
        weights /= weights.sum()
        return int(np.random.choice(len(results), p=weights))

    def generate_query(self, is_reformulation: bool = True) -> str:
        return f"{self.current_query} refined".strip()

    def record_action(self, action: str, target: str = "", result_title: str = "", is_ad: bool = False, reasoning: str = "", satisfaction: float = 0.0) -> None:
        super().record_action(action, target, result_title, is_ad, reasoning, satisfaction)
        if action == "next_page":
            self._pages_without_click += 1
        elif action == "click_result":
            self._pages_without_click = 0

    def reset(self) -> None:
        super().reset()
        self._pages_without_click = 0


class _PaperBaselineAgent(_BaseAgent):
    def __init__(self, persona: Persona, llm: Any, config: SearchAgentConfig):
        super().__init__(persona, config)
        self.llm = llm
        self._logical_calls = 0
        self._last_serp: Any = None

    def _payload(self, serp_description: Any) -> str:
        history = [
            {"step": e.step, "query": e.query, "action": e.action, "target": e.target, "result_title": e.result_title}
            for e in self.memory.entries[-8:]
        ]
        return json.dumps(
            {
                "persona": self.persona.to_dict(),
                "current_query": self.current_query,
                "visible_serp": serp_description,
                "recent_history": history,
                "session_meta": {"step": self.step_count, "clicks": self._total_clicks},
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _text_call(self, prompt: str, max_tokens: int = 256) -> str:
        if self._logical_calls >= 6:
            return ""
        self._logical_calls += 1
        return str(self.llm.generate_text(prompt, max_new_tokens=max_tokens) or "").strip()

    def _query_call(self, is_reformulation: bool) -> str:
        if self._logical_calls >= 6:
            return self.current_query
        self._logical_calls += 1
        mode = "reformulation" if is_reformulation else "new query"
        prompt = (
            f"Generate the user's {mode} from this search context. Preserve the broader goal and user language. "
            f"Return only the query JSON required by the interface.\nCONTEXT_JSON\n{self._payload(self._last_serp)}"
        )
        if hasattr(self.llm, "generate_query"):
            return str(self.llm.generate_query(prompt, self.current_query) or self.current_query).strip()
        text = self._text_call(prompt, 64)
        return text.splitlines()[0].strip().strip('"').strip("'") if text else self.current_query

    def _rank_call(self, serp_results: List[Any]) -> int:
        if self._logical_calls >= 6:
            return 0
        labels = [f"R{i + 1}" for i in range(len(serp_results))]
        candidates = []
        for index, result in enumerate(serp_results):
            item = result.to_prompt_dict() if hasattr(result, "to_prompt_dict") else {"rank": int(getattr(result, "rank", index + 1)), "title": str(getattr(result, "title", "")), "snippet": str(getattr(result, "snippet", "")), "url": str(getattr(result, "url", ""))}
            candidates.append({"label": labels[index], "result": item})
        prompt = (
            "Conditional on clicking, select the one visible result this baseline user would click. "
            "Use only the supplied visible fields.\n"
            f"CONTEXT_JSON\n{self._payload(self._last_serp)}\nCANDIDATES_JSON\n{json.dumps(candidates, ensure_ascii=False, sort_keys=True)}"
        )
        self._logical_calls += 1
        if hasattr(self.llm, "get_action_logprobs"):
            probs = self.llm.get_action_logprobs(prompt, labels)
            if probs:
                chosen = max(labels, key=lambda label: float(probs.get(label, 0.0)))
                return labels.index(chosen)
        return 0

    def _parse_with_retry(self, prompt: str, parser) -> _NativeDecision:
        for attempt in range(2):
            content = prompt if attempt == 0 else prompt + "\nReturn only one valid final machine-readable command in the required native format."
            text = self._text_call(content)
            decision = parser(text)
            if decision is not None and decision.action in MACRO_ACTIONS:
                return decision
        return _NativeDecision("stop_unsatisfied")

    def select_click_target(self, serp_description: Any, click_type: str, serp_results: Optional[list] = None) -> int:
        results = list(serp_results or [])
        if not results:
            return 0
        if self._pending_click_rank is not None:
            rank = self._pending_click_rank
            self._pending_click_rank = None
            for index, result in enumerate(results):
                if int(getattr(result, "rank", index + 1)) == rank:
                    return index
        return self._rank_call(results)

    def generate_query(self, is_reformulation: bool = True) -> str:
        if self._pending_query:
            query = self._pending_query
            self._pending_query = ""
            return query
        return self._query_call(is_reformulation)

    def reset(self) -> None:
        super().reset()
        self._logical_calls = 0
        self._last_serp = None


class NaiveLLMAgent(_PaperBaselineAgent):
    def __init__(self, persona: Persona, llm: Any, config: SearchAgentConfig):
        super().__init__(persona, llm, config)
        self._persona_desc = f"You are searching for: {persona.search_topic}"

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Return the next action label only as one of A, B, C, D, E, F, G. "
            "A=click_result, B=skip, C=next_page, D=reformulate, E=new_query, F=stop_satisfied, G=stop_unsatisfied.\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        text = self._text_call(prompt, 32).strip().upper()
        action = BaselineAdapter.cot(text[:1]) or "stop_unsatisfied"
        return self._set_decision(_NativeDecision(action))


class ReActSearchAgent(_PaperBaselineAgent):
    @staticmethod
    def _parse(text: str) -> Optional[_NativeDecision]:
        matches = re.findall(r"(?im)^\s*(?:FINAL\s+)?(?:ACTION|COMMAND)\s*:\s*(.+?)\s*$", text)
        command = matches[-1].strip() if matches else text.strip().splitlines()[-1].strip() if text.strip() else ""
        click = re.fullmatch(r"CLICK\((\d+)\)", command, re.I)
        if click:
            return _NativeDecision("click_result", click_rank=int(click.group(1)))
        if re.fullmatch(r"SKIP", command, re.I):
            return _NativeDecision("skip")
        if re.fullmatch(r"NEXT", command, re.I):
            return _NativeDecision("next_page")
        query = re.fullmatch(r"REFORMULATE\((.*)\)", command, re.I)
        if query and query.group(1).strip():
            return _NativeDecision("reformulate", query=query.group(1).strip().strip('"').strip("'"))
        query = re.fullmatch(r"NEW_QUERY\((.*)\)", command, re.I)
        if query and query.group(1).strip():
            return _NativeDecision("new_query", query=query.group(1).strip().strip('"').strip("'"))
        if re.fullmatch(r"STOP_SATISFIED", command, re.I):
            return _NativeDecision("stop_satisfied")
        if re.fullmatch(r"STOP_UNSATISFIED", command, re.I):
            return _NativeDecision("stop_unsatisfied")
        return None

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Use ReAct internally. Intermediate Thought text is internal computation. End with exactly one machine-readable line "
            "ACTION: CLICK(rank), ACTION: SKIP, ACTION: NEXT, ACTION: REFORMULATE(query), ACTION: NEW_QUERY(query), "
            "ACTION: STOP_SATISFIED, or ACTION: STOP_UNSATISFIED.\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


class PolySkillAgent(_PaperBaselineAgent):
    @staticmethod
    def _parse(text: str) -> Optional[_NativeDecision]:
        match = re.search(r"(?im)^\s*SKILL\s*:\s*([a-z_\-]+)(.*)$", text)
        if not match:
            return None
        skill = match.group(1).lower().replace("-", "_")
        args = match.group(2)
        rank_match = re.search(r"rank\s*=\s*(\d+)", args, re.I)
        query_match = re.search(r"query\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^;]+))", args, re.I)
        query = next((x.strip() for x in query_match.groups() if x and x.strip()), "") if query_match else ""
        if skill in {"click", "item_click"} and rank_match:
            return _NativeDecision("click_result", click_rank=int(rank_match.group(1)))
        if skill in {"navigation", "navigate"}:
            mode_match = re.search(r"mode\s*=\s*([a-z_\-]+)", args, re.I)
            mode = mode_match.group(1).lower().replace("-", "_") if mode_match else ""
            if mode in {"skip", "continue"}:
                return _NativeDecision("skip")
            if mode in {"next", "next_page"}:
                return _NativeDecision("next_page")
        if skill in {"query", "search", "query_change"} and query:
            mode_match = re.search(r"mode\s*=\s*([a-z_\-]+)", args, re.I)
            mode = mode_match.group(1).lower().replace("-", "_") if mode_match else ""
            if mode in {"refine", "refinement", "reformulate"}:
                return _NativeDecision("reformulate", query=query)
            if mode in {"new", "new_search", "new_query"}:
                return _NativeDecision("new_query", query=query)
        if skill in {"finish", "stop"}:
            status_match = re.search(r"status\s*=\s*([a-z_\-]+)", args, re.I)
            status = status_match.group(1).lower().replace("-", "_") if status_match else ""
            if status == "satisfied":
                return _NativeDecision("stop_satisfied")
            if status == "unsatisfied":
                return _NativeDecision("stop_unsatisfied")
        return None

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Select one PolySkill skill with associated arguments. Return exactly one line in one of these forms: "
            "SKILL: click rank=N; SKILL: navigation mode=skip; SKILL: navigation mode=next; "
            "SKILL: query mode=refine query=\"...\"; SKILL: query mode=new_search query=\"...\"; "
            "SKILL: finish status=satisfied; SKILL: finish status=unsatisfied.\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


class ReflexionAgent(ReActSearchAgent):
    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Perform the Reflexion method internally: reflect on the current search state, then produce one final command. "
            "Only the post-reflection command is executable. End with exactly one line ACTION: CLICK(rank), ACTION: SKIP, ACTION: NEXT, "
            "ACTION: REFORMULATE(query), ACTION: NEW_QUERY(query), ACTION: STOP_SATISFIED, or ACTION: STOP_UNSATISFIED.\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


class CoTAgent(_PaperBaselineAgent):
    @staticmethod
    def _parse(text: str) -> Optional[_NativeDecision]:
        label_matches = re.findall(r"(?im)^\s*(?:FINAL\s+)?(?:ACTION|LABEL)\s*:\s*([A-G])\s*$", text)
        if not label_matches:
            return None
        action = BaselineAdapter.cot(label_matches[-1])
        if not action:
            return None
        rank_match = re.findall(r"(?im)^\s*RANK\s*:\s*(\d+)\s*$", text)
        return _NativeDecision(action, click_rank=int(rank_match[-1]) if action == "click_result" and rank_match else None)

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Use Chain-of-Thought internally, then end with exactly one tagged final action label. "
            "A=click_result, B=skip, C=next_page, D=reformulate, E=new_query, F=stop_satisfied, G=stop_unsatisfied. "
            "The executable line must be ACTION: A through ACTION: G. If the label is A, you may also provide RANK: N as an auxiliary line.\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


class TreeOfThoughtsAgent(CoTAgent):
    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Use Tree of Thoughts internally with no more than six candidate branches. Score/select a leaf internally, then expose only the selected leaf's tagged final command. "
            "A=click_result, B=skip, C=next_page, D=reformulate, E=new_query, F=stop_satisfied, G=stop_unsatisfied. "
            "End with ACTION: A through ACTION: G. If the selected label is A, you may also provide RANK: N as an auxiliary line.\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


class UserSimCRSv2Agent(_PaperBaselineAgent):
    @staticmethod
    def _parse(text: str) -> Optional[_NativeDecision]:
        match = re.search(r"(?im)^\s*ACT\s*:\s*([a-z_\-]+)(.*)$", text)
        if not match:
            return None
        act = match.group(1).lower().replace("-", "_")
        args = match.group(2)
        rank_match = re.search(r"rank\s*=\s*(\d+)", args, re.I)
        sat_match = re.search(r"satisfaction\s*=\s*([01](?:\.\d+)?)", args, re.I)
        action = BaselineAdapter.usersimcrs(act, has_item=rank_match is not None, satisfaction=float(sat_match.group(1)) if sat_match else None)
        if not action:
            return None
        query_match = re.search(r"query\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|(.+))", args, re.I)
        query = next((x.strip() for x in query_match.groups() if x and x.strip()), "") if query_match else ""
        return _NativeDecision(action, click_rank=int(rank_match.group(1)) if rank_match else None, query=query if action == "reformulate" else "")

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Return one UserSimCRS v2 native act. Allowed forms: ACT: inspect rank=N; ACT: accept rank=N; ACT: reject; ACT: continue; "
            "ACT: request_more; ACT: clarify query=\"...\"; ACT: end satisfaction=X where X is in [0,1].\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


class CSHIAgent(_PaperBaselineAgent):
    @staticmethod
    def _parse(text: str) -> Optional[_NativeDecision]:
        match = re.search(r"(?im)^\s*ACT\s*:\s*([a-z_\-]+)(.*)$", text)
        if not match:
            return None
        act = match.group(1).lower().replace("-", "_")
        args = match.group(2)
        sat_match = re.search(r"satisfaction\s*=\s*([01](?:\.\d+)?)", args, re.I)
        action = BaselineAdapter.cshi(act, float(sat_match.group(1)) if sat_match else None)
        if not action:
            return None
        rank_match = re.search(r"rank\s*=\s*(\d+)", args, re.I)
        if action == "click_result" and not rank_match:
            return None
        query_match = re.search(r"query\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|(.+))", args, re.I)
        query = next((x.strip() for x in query_match.groups() if x and x.strip()), "") if query_match else ""
        if action in {"reformulate", "new_query"} and not query:
            return None
        return _NativeDecision(action, click_rank=int(rank_match.group(1)) if rank_match else None, query=query)

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Return one CSHI native act. Allowed forms: ACT: click rank=N; ACT: continue; ACT: refine query=\"...\"; "
            "ACT: switch query=\"...\"; ACT: finish satisfaction=X where X is in [0,1].\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


class USimAgent(_PaperBaselineAgent):
    def _parse(self, text: str) -> Optional[_NativeDecision]:
        match = re.search(r"(?im)^\s*EVENT\s*:\s*([a-z_\-]+)(.*)$", text)
        if not match:
            return None
        event = match.group(1).lower().replace("-", "_")
        args = match.group(2)
        rank_match = re.search(r"rank\s*=\s*(\d+)", args, re.I)
        mode_match = re.search(r"mode\s*=\s*([a-z_\-]+)", args, re.I)
        state_match = re.search(r"state\s*=\s*([a-z_\-]+)", args, re.I)
        query_match = re.search(r"query\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|(.+))", args, re.I)
        query = next((x.strip() for x in query_match.groups() if x and x.strip()), "") if query_match else ""
        action = BaselineAdapter.usimagent(event, self.current_query, query, mode_match.group(1) if mode_match else "", state_match.group(1) if state_match else "")
        if not action:
            return None
        if action == "click_result" and not rank_match:
            return None
        return _NativeDecision(action, click_rank=int(rank_match.group(1)) if rank_match else None, query=query if action in {"reformulate", "new_query"} else "")

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Return one USimAgent native event. Allowed forms: EVENT: click rank=N; EVENT: browse mode=skip; EVENT: browse mode=next_page; "
            "EVENT: query query=\"...\"; EVENT: stop state=satisfied; EVENT: stop state=unsatisfied. "
            "Query events are classified by normalized Levenshtein distance to the current query.\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


class BASESAgent(USimAgent):
    def __init__(self, persona: Persona, llm: Any, config: SearchAgentConfig):
        super().__init__(persona, llm, config)
        self._event_queue: List[dict] = []

    def _event_decision(self, event: dict) -> Optional[_NativeDecision]:
        name = str(event.get("event", ""))
        query = str(event.get("query", ""))
        action = BaselineAdapter.bases(name, self.current_query, query, str(event.get("mode", "")), event.get("terminal_utility"))
        if not action:
            return None
        rank = event.get("rank")
        if action == "click_result":
            try:
                rank = int(rank)
            except Exception:
                return None
        return _NativeDecision(action, click_rank=int(rank) if rank is not None else None, query=query if action in {"reformulate", "new_query"} else "")

    def _parse_sequence(self, text: str) -> Optional[List[dict]]:
        try:
            obj = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            try:
                obj = json.loads(match.group(0))
            except Exception:
                return None
        if not isinstance(obj, dict) or set(obj.keys()) != {"events"} or not isinstance(obj["events"], list) or not obj["events"]:
            return None
        events = []
        for raw in obj["events"]:
            if not isinstance(raw, dict):
                return None
            events.append(raw)
        return events

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        if not self._event_queue:
            prompt = (
                "Produce BASES's native whole-session event sequence as exactly one JSON object with sole key events. Each event is one of: "
                '{"event":"click","rank":N}, {"event":"browse","mode":"skip|next_page"}, {"event":"query","query":"..."}, '
                '{"event":"stop","terminal_utility":1..5}. Do not invent intermediate click events.\n'
                f"CONTEXT_JSON\n{self._payload(serp_description)}"
            )
            sequence = None
            for attempt in range(2):
                content = prompt if attempt == 0 else prompt + "\nReturn only the valid events JSON object."
                sequence = self._parse_sequence(self._text_call(content))
                if sequence:
                    break
            if not sequence:
                return self._set_decision(_NativeDecision("stop_unsatisfied"))
            self._event_queue = sequence
        event = self._event_queue.pop(0)
        decision = self._event_decision(event)
        if decision is None:
            self._event_queue = []
            return self._set_decision(_NativeDecision("stop_unsatisfied"))
        return self._set_decision(decision)

    def reset(self) -> None:
        super().reset()
        self._event_queue = []


class Agent4RecAgent(_PaperBaselineAgent):
    @staticmethod
    def _parse(text: str) -> Optional[_NativeDecision]:
        match = re.search(r"(?im)^\s*EVENT\s*:\s*([a-z_\-]+)(.*)$", text)
        if not match:
            return None
        event = match.group(1).lower().replace("-", "_")
        args = match.group(2)
        boundary_match = re.search(r"page_boundary\s*=\s*(true|false|1|0)", args, re.I)
        rating_match = re.search(r"rating\s*=\s*([a-z_\-]+)", args, re.I)
        action = BaselineAdapter.agent4rec(event, page_boundary=(boundary_match.group(1).lower() in {"true", "1"}) if boundary_match else None, exit_rating=rating_match.group(1) if rating_match else "")
        if not action:
            return None
        rank_match = re.search(r"rank\s*=\s*(\d+)", args, re.I)
        if action == "click_result" and not rank_match:
            return None
        query_match = re.search(r"query\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|(.+))", args, re.I)
        query = next((x.strip() for x in query_match.groups() if x and x.strip()), "") if query_match else ""
        if action in {"reformulate", "new_query"} and not query:
            return None
        return _NativeDecision(action, click_rank=int(rank_match.group(1)) if rank_match else None, query=query)

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Return one Agent4Rec native event. Allowed forms: EVENT: item_selection rank=N; EVENT: browsing page_boundary=true|false; "
            "EVENT: preference_refinement query=\"...\"; EVENT: goal_shift query=\"...\"; EVENT: exit rating=satisfied|unsatisfied.\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


class SimUSERAgent(_PaperBaselineAgent):
    @staticmethod
    def _parse(text: str) -> Optional[_NativeDecision]:
        match = re.search(r"(?im)^\s*ACT\s*:\s*([a-z_\-]+)(.*)$", text)
        if not match:
            return None
        act = match.group(1).lower().replace("-", "_")
        args = match.group(2)
        state_match = re.search(r"state\s*=\s*([a-z_\-]+)", args, re.I)
        action = BaselineAdapter.simuser(act, state_match.group(1) if state_match else "")
        if not action:
            return None
        rank_match = re.search(r"rank\s*=\s*(\d+)", args, re.I)
        if action == "click_result" and not rank_match:
            return None
        query_match = re.search(r"query\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|(.+))", args, re.I)
        query = next((x.strip() for x in query_match.groups() if x and x.strip()), "") if query_match else ""
        if action in {"reformulate", "new_query"} and not query:
            return None
        return _NativeDecision(action, click_rank=int(rank_match.group(1)) if rank_match else None, query=query)

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Return one SimUSER structured user act. Allowed forms: ACT: item_choice rank=N; ACT: continue; ACT: request_more; "
            "ACT: refine query=\"...\"; ACT: switch_goal query=\"...\"; ACT: finish state=satisfied|unsatisfied.\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


class ConvSearchR1Agent(_PaperBaselineAgent):
    def _parse(self, text: str) -> Optional[_NativeDecision]:
        match = re.search(r"(?im)^\s*ACT\s*:\s*([a-z_\-]+)(.*)$", text)
        if not match:
            return None
        act = match.group(1).lower().replace("-", "_")
        args = match.group(2)
        if act == "browse":
            return _NativeDecision("skip")
        if act == "click":
            rank_match = re.search(r"rank\s*=\s*(\d+)", args, re.I)
            return _NativeDecision("click_result", click_rank=int(rank_match.group(1))) if rank_match else None
        if act == "finish":
            state_match = re.search(r"state\s*=\s*([a-z_\-]+)", args, re.I)
            state = state_match.group(1).lower() if state_match else ""
            if state == "satisfied":
                return _NativeDecision("stop_satisfied")
            if state == "unsatisfied":
                return _NativeDecision("stop_unsatisfied")
            return None
        if act in {"search", "reformulate"}:
            query_match = re.search(r"query\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|(.+))", args, re.I)
            query = next((x.strip() for x in query_match.groups() if x and x.strip()), "") if query_match else ""
            action = BaselineAdapter.convsearch(self.current_query, query)
            return _NativeDecision(action, query=query) if action else None
        return None

    def decide_action(self, serp_description: Any) -> Tuple[str, Dict[str, float]]:
        self._logical_calls = 0
        self._last_serp = serp_description
        prompt = (
            "Return one ConvSearch-R1 native act. Allowed forms: ACT: search query=\"...\"; ACT: reformulate query=\"...\"; "
            "ACT: browse; ACT: click rank=N; ACT: finish state=satisfied|unsatisfied. Generated queries are categorized by canonical-token retention.\n"
            f"CONTEXT_JSON\n{self._payload(serp_description)}"
        )
        return self._set_decision(self._parse_with_retry(prompt, self._parse))


DualPromptLLMAgent = NaiveLLMAgent
UXSimHybridAgent = CSHIAgent
SkillsBasedAgent = PolySkillAgent
FewShotDemoAgent = UserSimCRSv2Agent
ControllablePersonaAgent = NaiveLLMAgent
SelfReflectionAgent = ReflexionAgent
ValueGuidedAgent = TreeOfThoughtsAgent
