"""Agent memory module for tracking search session history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    """A single memory entry from a search interaction."""

    step: int
    query: str
    action: str
    target: str = ""
    result_title: str = ""
    is_ad: bool = False
    satisfaction: float = 0.0
    reasoning: str = ""

    def to_text(self) -> str:
        parts = [f"Step {self.step}: query='{self.query}', action={self.action}"]
        if self.target:
            parts.append(f"target='{self.target}'")
        if self.is_ad:
            parts.append("(advertisement)")
        if self.satisfaction > 0 and self.action in ("click_organic", "click_ad"):
            sat_label = {1: "very unsatisfied", 2: "unsatisfied", 3: "neutral",
                         4: "satisfied", 5: "very satisfied"}.get(
                round(self.satisfaction), f"satisfaction={self.satisfaction:.1f}")
            parts.append(f"[{sat_label}]")
        if self.reasoning:
            parts.append(f"reason: {self.reasoning}")
        return ", ".join(parts)


class AgentMemory:
    """Manages the search session memory for an agent.

    Stores interaction history and provides context for LLM prompts.
    Inspired by RecAgent's memory mechanism but adapted for search sessions.
    """

    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.queries_issued: List[str] = []
        self.clicks: List[Dict[str, Any]] = []
        self.ad_clicks: List[Dict[str, Any]] = []
        self.pages_viewed: int = 0
        self.total_skips: int = 0

    def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry."""
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        if entry.action in ("reformulate", "new_query"):
            self.queries_issued.append(entry.query)
        elif entry.action == "click_organic":
            self.clicks.append({"step": entry.step, "target": entry.target, "is_ad": False})
        elif entry.action == "click_ad":
            self.clicks.append({"step": entry.step, "target": entry.target, "is_ad": True})
            self.ad_clicks.append({"step": entry.step, "target": entry.target})
        elif entry.action == "next_page":
            self.pages_viewed += 1
        elif entry.action == "skip":
            self.total_skips += 1

    def get_recent_context(self, n: int = 5) -> str:
        """Get recent memory entries as text for LLM context."""
        recent = self.entries[-n:] if len(self.entries) > n else self.entries
        if not recent:
            return "No previous actions in this session."
        lines = ["Recent actions in this session:"]
        for entry in recent:
            lines.append(f"  - {entry.to_text()}")
        return "\n".join(lines)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        return {
            "total_steps": len(self.entries),
            "queries_issued": len(self.queries_issued),
            "organic_clicks": len(self.clicks) - len(self.ad_clicks),
            "ad_clicks": len(self.ad_clicks),
            "pages_viewed": self.pages_viewed,
            "total_skips": self.total_skips,
        }

    def reset(self) -> None:
        """Reset memory for a new session."""
        self.entries.clear()
        self.queries_issued.clear()
        self.clicks.clear()
        self.ad_clicks.clear()
        self.pages_viewed = 0
        self.total_skips = 0
