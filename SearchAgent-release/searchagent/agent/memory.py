from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MemoryEntry:
    step: int
    query: str
    action: str
    target: str = ""
    result_title: str = ""
    is_ad: bool = False
    satisfaction: float = 0.0
    dwell_time: float = 0.0
    reasoning: str = ""

    def to_text(self) -> str:
        parts = [f"step={self.step}", f"query={self.query!r}", f"action={self.action}"]
        if self.target:
            parts.append(f"target={self.target!r}")
        if self.satisfaction > 0:
            parts.append(f"utility={self.satisfaction:.3f}")
        if self.dwell_time > 0:
            parts.append(f"dwell={self.dwell_time:.1f}")
        return ", ".join(parts)


class AgentMemory:
    def __init__(self, max_entries: int = 8):
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.queries_issued: List[str] = []
        self.clicks: List[Dict[str, Any]] = []
        self.ad_clicks: List[Dict[str, Any]] = []
        self.pages_viewed = 0
        self.total_skips = 0

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        if entry.action in ("reformulate", "new_query"):
            self.queries_issued.append(entry.target or entry.query)
        elif entry.action in ("click_organic", "click_ad", "click_result"):
            item = {"step": entry.step, "target": entry.target, "is_ad": entry.is_ad}
            self.clicks.append(item)
            if entry.is_ad or entry.action == "click_ad":
                self.ad_clicks.append(item)
        elif entry.action == "next_page":
            self.pages_viewed += 1
        elif entry.action == "skip":
            self.total_skips += 1

    def get_recent_context(self, n: int = 8) -> str:
        recent = self.entries[-n:]
        if not recent:
            return "No previous actions in this session."
        return "\n".join(entry.to_text() for entry in recent)

    def get_recent_records(self, n: int = 8) -> List[Dict[str, Any]]:
        records = []
        for entry in self.entries[-n:]:
            records.append({
                "step": entry.step,
                "query": entry.query,
                "action": entry.action,
                "target": entry.target,
                "utility": entry.satisfaction,
            })
        return records

    def get_session_summary(self) -> Dict[str, Any]:
        return {
            "total_steps": len(self.entries),
            "queries_issued": len(self.queries_issued),
            "clicks": len(self.clicks),
            "ad_clicks": len(self.ad_clicks),
            "pages_viewed": self.pages_viewed,
            "total_skips": self.total_skips,
        }

    def reset(self) -> None:
        self.entries.clear()
        self.queries_issued.clear()
        self.clicks.clear()
        self.ad_clicks.clear()
        self.pages_viewed = 0
        self.total_skips = 0
