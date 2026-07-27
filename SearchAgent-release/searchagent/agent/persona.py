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
    """Extract persona traits from real TREC user behavior.

    Instead of randomly assigning traits, we infer them from observable
    behavioral signals in the TREC session data (analogous to Agent4Rec's
    approach of initializing agent profiles from real user history).

    Trait inference rules (based on TREC Session Track statistics):
      - patience_level:  session length & num interactions
      - search_expertise: query reformulation quality & click depth
      - decision_style:  click speed (rank of first click) & num clicks
      - ad_sensitivity:  proportion of ad-URL clicks (limited in TREC)
    """

    # Thresholds for patience inference.
    # Uses engagement score = n_interactions + n_clicks + dwell_bonus
    # to avoid mislabeling single-interaction, high-click sessions as "low patience".
    PATIENCE_ENGAGEMENT_LOW = 2    # engagement score <= 2 → low patience
    PATIENCE_ENGAGEMENT_HIGH = 5   # engagement score > 5 → high patience
    EXPERTISE_REFORMULATION_THRESHOLD = 2  # sessions with >=2 reformulations
    CLICK_DEPTH_EXPERT = 3  # clicking rank>=3 suggests deliberate scanning

    @classmethod
    def extract_from_session(cls, trec_session) -> dict:
        """Infer trait values from a single TREC session.

        Returns dict with keys: patience_level, search_expertise,
        decision_style, ad_sensitivity.
        """
        n_interactions = len(trec_session.interactions)
        n_clicks = trec_session.total_clicks
        click_ranks = trec_session.click_positions
        has_clicks = trec_session.has_clicks
        total_dwell = trec_session.total_dwell_time

        # ── patience_level ──
        # Engagement score combines interactions, clicks, and dwell time.
        # A user with 1 interaction but 5 clicks is engaged (NOT impatient).
        dwell_bonus = 1 if total_dwell > 60 else 0  # >60s total dwell = engaged
        engagement = n_interactions + n_clicks + dwell_bonus
        if engagement <= cls.PATIENCE_ENGAGEMENT_LOW:
            patience = "low"
        elif engagement <= cls.PATIENCE_ENGAGEMENT_HIGH:
            patience = "medium"
        else:
            patience = "high"

        # ── search_expertise ──
        # Experts: reformulate queries (>1 interaction) AND click beyond rank 1
        has_deep_clicks = any(r >= cls.CLICK_DEPTH_EXPERT for r in click_ranks) if click_ranks else False
        n_reformulations = max(0, n_interactions - 1)
        if n_reformulations >= cls.EXPERTISE_REFORMULATION_THRESHOLD and has_deep_clicks:
            expertise = "expert"
        elif n_reformulations >= 1 or has_deep_clicks:
            expertise = "intermediate"
        else:
            expertise = "novice"

        # ── decision_style ──
        # Impulsive: clicks only rank 1, many clicks relative to interactions
        # Deliberate: diverse click ranks, longer dwell times
        if click_ranks:
            mean_rank = sum(click_ranks) / len(click_ranks)
            rank_variety = len(set(click_ranks))
            avg_dwell = total_dwell / len(click_ranks) if total_dwell > 0 else 0
        else:
            mean_rank = 0
            rank_variety = 0
            avg_dwell = 0

        if n_clicks > 0 and mean_rank <= 1.5 and rank_variety <= 1:
            style = "impulsive"
        elif avg_dwell > 30 or rank_variety >= 3:
            style = "deliberate"
        else:
            style = "deliberate" if mean_rank > 2 else "impulsive"

        # ── ad_sensitivity ──
        # TREC has very few ads; use URL-based heuristic from serp.py
        ad_clicks = 0
        organic_clicks = 0
        for inter in trec_session.interactions:
            for click in inter.clicks:
                # Match click rank to result
                matched = [r for r in inter.results if r.rank == click.rank]
                if matched and matched[0].is_ad:
                    ad_clicks += 1
                else:
                    organic_clicks += 1

        if ad_clicks > 0:
            ad_ratio = ad_clicks / max(ad_clicks + organic_clicks, 1)
            if ad_ratio > 0.3:
                ad_sens = "ad_tolerant"
            else:
                ad_sens = "ad_neutral"
        else:
            ad_sens = "ad_averse" if n_clicks >= 2 else "ad_neutral"

        return {
            "patience_level": patience,
            "search_expertise": expertise,
            "decision_style": style,
            "ad_sensitivity": ad_sens,
        }

    @classmethod
    def extract_from_user_sessions(cls, sessions: list) -> dict:
        """Aggregate traits across multiple sessions from the same user.

        Uses majority vote across sessions for each trait.
        """
        from collections import Counter
        trait_votes: Dict[str, list] = {
            "patience_level": [], "search_expertise": [],
            "decision_style": [], "ad_sensitivity": [],
        }
        for s in sessions:
            traits = cls.extract_from_session(s)
            for k, v in traits.items():
                trait_votes[k].append(v)

        result = {}
        for k, votes in trait_votes.items():
            if votes:
                result[k] = Counter(votes).most_common(1)[0][0]
            else:
                result[k] = {"patience_level": "medium", "search_expertise": "intermediate",
                             "decision_style": "deliberate", "ad_sensitivity": "ad_neutral"}[k]
        return result

    @classmethod
    def create_persona(cls, trec_session, persona_id: int = 0) -> Persona:
        """Create a Persona from a single TREC session's behavior."""
        traits = cls.extract_from_session(trec_session)
        return Persona(
            persona_id=persona_id,
            age=30,  # default; TREC has no demographics
            search_expertise=traits["search_expertise"],
            ad_sensitivity=traits["ad_sensitivity"],
            decision_style=traits["decision_style"],
            patience_level=traits["patience_level"],
            search_topic=trec_session.interactions[0].query if trec_session.interactions else "",
            search_goal=trec_session.topic_desc[:200] if trec_session.topic_desc else "",
        )
