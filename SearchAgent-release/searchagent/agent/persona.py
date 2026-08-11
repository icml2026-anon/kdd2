"""Persona generation and management for search user agents."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from searchagent.config import PersonaConfig


@dataclass
class Persona:
    """A user persona that drives LLM agent behavior."""

    persona_id: int
    age: int
    search_expertise: str
    ad_sensitivity: str
    decision_style: str
    patience_level: str
    search_topic: str = ""
    search_goal: str = ""
    background: str = ""

    def to_prompt_description(self) -> str:
        """Generate a natural language description for LLM prompts."""
        # Natural language descriptions instead of raw labels
        expertise_desc = {
            "novice": "You are not very experienced with search engines and often struggle to find what you need.",
            "intermediate": "You have moderate experience with search engines.",
            "expert": "You are highly skilled at searching and know how to craft precise queries.",
        }
        ad_desc = {
            "ad_tolerant": "You are open to clicking on advertisements if they look relevant to your needs.",
            "ad_neutral": "You sometimes click ads if they seem useful, but generally prefer organic results.",
            "ad_averse": "You strongly prefer organic results and almost never click on advertisements.",
        }
        style_desc = {
            "impulsive": "You tend to click quickly on the first promising result without much deliberation.",
            "deliberate": "You carefully compare multiple results before clicking, reading snippets thoroughly.",
            "price_sensitive": "You are very focused on finding the best deal and compare prices across results.",
        }
        patience_desc = {
            "low": "You have little patience — if results aren't immediately useful, you leave quickly.",
            "medium": "You have moderate patience and will try a few different approaches before giving up.",
            "high": "You are very patient and willing to browse many results and try multiple queries.",
        }

        lines = [f"You are a {self.age}-year-old user searching the web."]
        lines.append(expertise_desc.get(self.search_expertise, f"Search skill: {self.search_expertise}."))
        lines.append(ad_desc.get(self.ad_sensitivity, f"Ad attitude: {self.ad_sensitivity}."))
        lines.append(style_desc.get(self.decision_style, f"Style: {self.decision_style}."))
        lines.append(patience_desc.get(self.patience_level, f"Patience: {self.patience_level}."))
        if self.background:
            lines.append(f"Background: {self.background}")
        if self.search_topic:
            lines.append(f"You are currently searching for: {self.search_topic}")
        if self.search_goal:
            lines.append(f"Your goal: {self.search_goal}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "persona_id": self.persona_id,
            "age": self.age,
            "search_expertise": self.search_expertise,
            "ad_sensitivity": self.ad_sensitivity,
            "decision_style": self.decision_style,
            "patience_level": self.patience_level,
            "search_topic": self.search_topic,
            "search_goal": self.search_goal,
            "background": self.background,
        }


# Search topics with goals for simulation
SEARCH_SCENARIOS = [
    {
        "topic": "best noise cancelling headphones",
        "goal": "Find and possibly purchase headphones under $300",
        "background": "You have been researching audio equipment for a week",
    },
    {
        "topic": "cheap flights to Tokyo",
        "goal": "Book the cheapest round-trip flight for a vacation next month",
        "background": "You are price-sensitive and compare multiple airlines",
    },
    {
        "topic": "python machine learning tutorial",
        "goal": "Find a comprehensive beginner-friendly tutorial to learn ML",
        "background": "You are a software developer expanding your skills",
    },
    {
        "topic": "best running shoes 2024",
        "goal": "Find running shoes suitable for marathon training",
        "background": "You are an amateur runner preparing for your first marathon",
    },
    {
        "topic": "home insurance quotes comparison",
        "goal": "Get the best deal on home insurance renewal",
        "background": "Your current policy expires in two weeks",
    },
    {
        "topic": "Italian restaurant near me",
        "goal": "Find a good Italian restaurant for a dinner reservation tonight",
        "background": "You are looking for a place for a birthday dinner",
    },
    {
        "topic": "laptop for video editing under 1500",
        "goal": "Purchase a laptop powerful enough for 4K video editing",
        "background": "You are a freelance video creator upgrading your equipment",
    },
    {
        "topic": "symptoms of vitamin D deficiency",
        "goal": "Understand whether your symptoms match vitamin D deficiency",
        "background": "You have been feeling tired and want to self-diagnose before seeing a doctor",
    },
    {
        "topic": "best credit card cashback rewards",
        "goal": "Apply for a credit card with the best cashback program",
        "background": "You currently have no rewards card and spend heavily on groceries",
    },
    {
        "topic": "online MBA programs ranking",
        "goal": "Shortlist top 3 affordable online MBA programs",
        "background": "You are a working professional considering further education",
    },
]


class PersonaGenerator:
    """Generate diverse user personas for simulation."""

    def __init__(self, config: PersonaConfig, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)

    def generate(self, n: int, scenarios: Optional[List[Dict]] = None) -> List[Persona]:
        """Generate n diverse personas with search scenarios."""
        if scenarios is None:
            scenarios = SEARCH_SCENARIOS

        personas = []
        for i in range(n):
            scenario = scenarios[i % len(scenarios)]
            persona = Persona(
                persona_id=i,
                age=self.rng.randint(*self.config.age_range),
                search_expertise=self.rng.choice(self.config.search_expertise),
                ad_sensitivity=self.rng.choice(self.config.ad_sensitivity),
                decision_style=self.rng.choice(self.config.decision_style),
                patience_level=self.rng.choice(self.config.patience_level),
                search_topic=scenario["topic"],
                search_goal=scenario["goal"],
                background=scenario["background"],
            )
            personas.append(persona)
        return personas

    def generate_one(self, persona_id: int = 0) -> Persona:
        """Generate a single persona."""
        return self.generate(1)[0]


class TRECPersonaExtractor:
    @staticmethod
    def _initial_features(trec_session):
        if not trec_session.interactions:
            return 0.5, 0.5, 0.5
        first = trec_session.interactions[0]
        query_tokens = [token for token in first.query.lower().split() if token]
        query_set = set(query_tokens)
        overlaps = []
        for result in first.results:
            text = f"{getattr(result, 'title', '')} {getattr(result, 'snippet', '')}".lower().split()
            result_set = set(text)
            overlaps.append(len(query_set & result_set) / max(len(query_set | result_set), 1))
        mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        specificity = min(1.0, len(query_tokens) / 8.0)
        if any(any(ch.isdigit() for ch in token) for token in query_tokens):
            specificity = min(1.0, specificity + 0.1)
        if query_tokens and query_tokens[0] in {"what", "why", "how", "when", "where", "who", "which"}:
            specificity = min(1.0, specificity + 0.1)
        concentration = min(1.0, len(first.results) / 10.0)
        return specificity, mean_overlap, concentration

    @classmethod
    def extract_from_session(cls, trec_session) -> dict:
        specificity, overlap, concentration = cls._initial_features(trec_session)
        expertise_score = min(1.0, 0.65 * specificity + 0.35 * overlap)
        style_score = min(1.0, 0.55 * concentration + 0.45 * (1.0 - overlap))
        patience_score = min(1.0, 0.5 * concentration + 0.5 * (1.0 - specificity))
        expertise = "novice" if expertise_score < 0.33 else "expert" if expertise_score > 0.67 else "intermediate"
        style = "impulsive" if style_score < 0.4 else "deliberate"
        patience = "low" if patience_score < 0.33 else "high" if patience_score > 0.67 else "medium"
        return {
            "patience_level": patience,
            "search_expertise": expertise,
            "decision_style": style,
            "ad_sensitivity": "ad_neutral",
        }

    @classmethod
    def extract_from_user_sessions(cls, sessions: list) -> dict:
        from collections import Counter
        values = [cls.extract_from_session(session) for session in sessions]
        defaults = {
            "patience_level": "medium",
            "search_expertise": "intermediate",
            "decision_style": "deliberate",
            "ad_sensitivity": "ad_neutral",
        }
        return {key: Counter([value[key] for value in values]).most_common(1)[0][0] if values else default for key, default in defaults.items()}

    @classmethod
    def create_persona(cls, trec_session, persona_id: int = 0) -> Persona:
        traits = cls.extract_from_session(trec_session)
        return Persona(
            persona_id=persona_id,
            age=30,
            search_expertise=traits["search_expertise"],
            ad_sensitivity=traits["ad_sensitivity"],
            decision_style=traits["decision_style"],
            patience_level=traits["patience_level"],
            search_topic=trec_session.interactions[0].query if trec_session.interactions else "",
            search_goal=trec_session.topic_desc[:200] if trec_session.topic_desc else "",
        )
