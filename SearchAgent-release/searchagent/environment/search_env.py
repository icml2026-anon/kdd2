"""Search environment that provides SERP pages and processes agent actions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from searchagent.config import EnvironmentConfig
from searchagent.environment.serp import SERPGenerator, SERPPage, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ClickRecord:
    """Record of a click event."""

    step: int
    query: str
    rank: int
    title: str
    is_ad: bool
    relevance: float
    satisfaction: float = 0.0  # 1-5 LLM-rated satisfaction
    dwell_time: float = 0.0   # estimated dwell time in seconds


@dataclass
class SessionRecord:
    """Complete record of a simulated search session."""

    persona_id: int
    initial_query: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    clicks: List[ClickRecord] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    total_steps: int = 0
    abandoned: bool = False

    @property
    def session_length(self) -> int:
        return self.num_queries

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def organic_clicks(self) -> int:
        return sum(1 for c in self.clicks if not c.is_ad)

    @property
    def ad_clicks(self) -> int:
        return sum(1 for c in self.clicks if c.is_ad)

    @property
    def click_positions(self) -> List[int]:
        return [c.rank for c in self.clicks]

    @property
    def mean_satisfaction(self) -> float:
        if not self.clicks:
            return 0.0
        sats = [c.satisfaction for c in self.clicks if c.satisfaction > 0]
        return float(np.mean(sats)) if sats else 0.0

    @property
    def total_dwell_time(self) -> float:
        return sum(c.dwell_time for c in self.clicks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "initial_query": self.initial_query,
            "total_steps": self.total_steps,
            "num_queries": self.num_queries,
            "organic_clicks": self.organic_clicks,
            "ad_clicks": self.ad_clicks,
            "abandoned": self.abandoned,
            "click_positions": self.click_positions,
            "queries": self.queries,
            "steps": self.steps,
            "clicks": [
                {
                    "step": c.step, "query": c.query, "rank": c.rank,
                    "title": c.title, "is_ad": c.is_ad, "relevance": c.relevance,
                    "satisfaction": c.satisfaction, "dwell_time": c.dwell_time,
                }
                for c in self.clicks
            ],
            "mean_satisfaction": self.mean_satisfaction,
            "total_dwell_time": self.total_dwell_time,
        }


class SearchEnvironment:
    """Search environment that manages SERP generation and interaction tracking."""

    def __init__(self, config: EnvironmentConfig, seed: int = 42):
        self.config = config
        self.serp_gen = SERPGenerator(config, seed=seed)
        self.current_serp: Optional[SERPPage] = None
        self.current_page: int = 1
        self.session_record: Optional[SessionRecord] = None

    def reset(self, persona_id: int, initial_query: str) -> SERPPage:
        """Reset environment for a new session."""
        self.current_page = 1
        self.current_serp = self.serp_gen.generate(initial_query, self.current_page)
        self.session_record = SessionRecord(
            persona_id=persona_id,
            initial_query=initial_query,
            queries=[initial_query],
        )
        return self.current_serp

    @classmethod
    def from_trec_session(cls, trec_session, config: EnvironmentConfig) -> "ReplayEnvironment":
        """Create a ReplayEnvironment from a TREC session."""
        return ReplayEnvironment(trec_session, config)

    def step(
        self,
        action: str,
        click_index: Optional[int] = None,
        new_query: Optional[str] = None,
    ) -> Tuple[Optional[SERPPage], bool]:
        """Execute an action and return the new SERP (if any) and done flag.

        Args:
            action: The macro-action to execute.
            click_index: Index of the result to click (for click actions).
            new_query: The new query string (for reformulate/new_query).

        Returns:
            Tuple of (new_serp_or_none, is_session_done)
        """
        assert self.session_record is not None, "Call reset() first"
        step_info = {"action": action, "step": self.session_record.total_steps}

        if action in ("abandon", "stop_unsatisfied", "end_session", "stop_satisfied"):
            self.session_record.abandoned = (len(self.session_record.clicks) == 0)
            self.session_record.steps.append(step_info)
            self.session_record.total_steps += 1
            return None, True

        elif action in ("click_organic", "click_ad", "click_result"):
            if click_index is not None and self.current_serp is not None:
                results = self.current_serp.results
                if 0 <= click_index < len(results):
                    result = results[click_index]
                    record = ClickRecord(
                        step=self.session_record.total_steps,
                        query=self.current_serp.query,
                        rank=result.rank,
                        title=result.title,
                        is_ad=result.is_ad,
                        relevance=result.relevance,
                    )
                    self.session_record.clicks.append(record)
                    step_info["rank"] = result.rank
                    step_info["is_ad"] = result.is_ad
                    step_info["title"] = result.title

            self.session_record.steps.append(step_info)
            self.session_record.total_steps += 1
            return self.current_serp, False

        elif action == "skip":
            self.session_record.steps.append(step_info)
            self.session_record.total_steps += 1
            return self.current_serp, False

        elif action == "next_page":
            self.current_page += 1
            query = self.current_serp.query if self.current_serp else ""
            self.current_serp = self.serp_gen.generate(query, self.current_page)
            step_info["page"] = self.current_page
            self.session_record.steps.append(step_info)
            self.session_record.total_steps += 1
            return self.current_serp, False

        elif action in ("reformulate", "new_query"):
            if new_query:
                self.current_page = 1
                self.current_serp = self.serp_gen.generate(new_query, self.current_page)
                self.session_record.queries.append(new_query)
                step_info["new_query"] = new_query
            self.session_record.steps.append(step_info)
            self.session_record.total_steps += 1
            return self.current_serp, False

        else:
            logger.warning("Unknown action: %s", action)
            self.session_record.steps.append(step_info)
            self.session_record.total_steps += 1
            return self.current_serp, False

    def update_last_click(self, satisfaction: float, dwell_time: float) -> None:
        """Attach satisfaction and dwell time to the most recent click."""
        if self.session_record and self.session_record.clicks:
            self.session_record.clicks[-1].satisfaction = satisfaction
            self.session_record.clicks[-1].dwell_time = dwell_time

    def get_session_record(self) -> SessionRecord:
        """Return the current session record."""
        assert self.session_record is not None
        return self.session_record


class ReplayEnvironment(SearchEnvironment):
    """Environment that replays real TREC SERPs instead of generating synthetic ones.

    The agent sees the exact same SERP pages that the real user saw.
    Ground truth (real clicks, dwell times) is stored for comparison.
    When the agent reformulates, the next TREC interaction's SERP is served.

    serp_perturbation controls result ordering:
      'none'    — original TREC ranking (default)
      'shuffle' — random permutation of results
      'reverse' — reverse the original ranking (worst first)
    """

    def __init__(self, trec_session, config: EnvironmentConfig,
                 serp_perturbation: str = 'none', rng=None):
        # Don't call super().__init__ — we don't need SERPGenerator
        self.config = config
        self.trec_session = trec_session
        self._interaction_idx: int = 0
        self._serp_perturbation = serp_perturbation
        self._rng = rng or np.random.RandomState(42)
        self.current_serp: Optional[SERPPage] = None
        self.current_page: int = 1
        self.session_record: Optional[SessionRecord] = None

        # Ground truth from real user
        self.ground_truth_clicks = []  # [(interaction_idx, rank, dwell)]
        self.ground_truth_queries = []
        for idx, inter in enumerate(trec_session.interactions):
            self.ground_truth_queries.append(inter.query)
            for c in inter.clicks:
                self.ground_truth_clicks.append((idx, c.rank, c.dwell_time))

    def _perturb_serp(self, serp: SERPPage) -> SERPPage:
        """Apply perturbation to SERP result ordering."""
        if self._serp_perturbation == 'none' or not serp.results:
            return serp
        results = list(serp.results)
        if self._serp_perturbation == 'shuffle':
            self._rng.shuffle(results)
            for i, r in enumerate(results):
                r.rank = i + 1
        elif self._serp_perturbation == 'reverse':
            results = results[::-1]
            for i, r in enumerate(results):
                r.rank = i + 1
        return SERPPage(query=serp.query, page_number=serp.page_number, results=results)

    def reset(self, persona_id: int, initial_query: str) -> SERPPage:
        """Reset with the first real SERP from the TREC session."""
        self._interaction_idx = 0
        self.current_page = 1
        first_inter = self.trec_session.interactions[0]
        self.current_serp = self._perturb_serp(first_inter.to_serp())
        self.session_record = SessionRecord(
            persona_id=persona_id,
            initial_query=first_inter.query or initial_query,
            queries=[first_inter.query or initial_query],
        )
        return self.current_serp

    def step(
        self,
        action: str,
        click_index: Optional[int] = None,
        new_query: Optional[str] = None,
    ) -> Tuple[Optional[SERPPage], bool]:
        """Process action; for reformulate/new_query, serve next TREC SERP."""
        assert self.session_record is not None
        step_info = {"action": action, "step": self.session_record.total_steps}

        if action in ("abandon", "end_session", "stop_satisfied", "stop_unsatisfied"):
            self.session_record.abandoned = (len(self.session_record.clicks) == 0)
            self.session_record.steps.append(step_info)
            self.session_record.total_steps += 1
            return None, True

        elif action in ("click_organic", "click_ad", "click_result"):
            if click_index is not None and self.current_serp is not None:
                results = self.current_serp.results
                if 0 <= click_index < len(results):
                    result = results[click_index]
                    record = ClickRecord(
                        step=self.session_record.total_steps,
                        query=self.current_serp.query,
                        rank=result.rank,
                        title=result.title,
                        is_ad=result.is_ad,
                        relevance=result.relevance,
                    )
                    self.session_record.clicks.append(record)
                    step_info["rank"] = result.rank
                    step_info["is_ad"] = result.is_ad
            self.session_record.steps.append(step_info)
            self.session_record.total_steps += 1
            return self.current_serp, False

        elif action in ("reformulate", "new_query"):
            # Advance to next TREC interaction's SERP (if available)
            self._interaction_idx += 1
            if self._interaction_idx < len(self.trec_session.interactions):
                next_inter = self.trec_session.interactions[self._interaction_idx]
                self.current_serp = self._perturb_serp(next_inter.to_serp())
                q = new_query or next_inter.query
                self.session_record.queries.append(q)
                step_info["new_query"] = q
            else:
                # No more TREC SERPs available — session should end
                step_info["new_query"] = new_query or ""
                self.session_record.steps.append(step_info)
                self.session_record.total_steps += 1
                return self.current_serp, True  # force done

            self.session_record.steps.append(step_info)
            self.session_record.total_steps += 1
            return self.current_serp, False

        else:  # skip, next_page, etc.
            self.session_record.steps.append(step_info)
            self.session_record.total_steps += 1
            return self.current_serp, False
