"""Prompt templates for search user agent interactions."""

from __future__ import annotations

from typing import Dict, List, Optional


# Action labels for forced-choice selection
ACTION_LABELS = ["A", "B", "C", "D", "E", "F", "G"]

ACTION_DESCRIPTIONS = {
    "click_result": "Click on a search result (either organic or sponsored)",
    "skip": "Skip the current results and continue scanning",
    "next_page": "Go to the next page of search results",
    "reformulate": "Reformulate/modify the current search query",
    "new_query": "Start a completely new search query",
    "stop_satisfied": "Done — you found what you needed and leave satisfied",
    "stop_unsatisfied": "Give up — the results are not helpful at all",
}

MACRO_ACTIONS = list(ACTION_DESCRIPTIONS.keys())


def build_action_selection_prompt(
    persona_desc: str,
    current_query: str,
    serp_description: str,
    memory_context: str,
    intent_description: str,
    css_context: str = "",
) -> str:
    """Build a forced-choice prompt for macro-action selection.

    The LLM picks one of A-G; we extract logprobs to get P(action).
    """
    action_lines = []
    for i, (action, desc) in enumerate(ACTION_DESCRIPTIONS.items()):
        action_lines.append(f"{ACTION_LABELS[i]}. {desc}")
    actions_text = "\n".join(action_lines)

    css_block = f"\nCognitive state: {css_context}\n" if css_context else ""

    prompt = f"""{persona_desc}

Current search query: "{current_query}"
Current intent: {intent_description}
{css_block}
{memory_context}

Search results page:
{serp_description}

Based on your persona, search history, and the results shown above, what would you do next?
Choose exactly one action (respond with ONLY the letter):
{actions_text}

Answer:"""
    return prompt


def build_query_generation_prompt(
    persona_desc: str,
    original_query: str,
    memory_context: str,
    intent_description: str,
    is_reformulation: bool = True,
) -> str:
    """Build a prompt for generating a new or reformulated query."""
    if is_reformulation:
        instruction = (
            f'Your original query was: "{original_query}"\n'
            "The results were not satisfactory. Reformulate the query to get better results.\n"
            "Generate ONLY the new query text, nothing else."
        )
    else:
        instruction = (
            f'Your previous query was: "{original_query}"\n'
            "You want to explore a different aspect of your search goal.\n"
            "Generate ONLY the new query text, nothing else."
        )

    prompt = f"""{persona_desc}

{memory_context}

Current intent: {intent_description}

{instruction}

New query:"""
    return prompt


def build_click_selection_prompt(
    persona_desc: str,
    current_query: str,
    serp_description: str,
    memory_context: str,
    click_type: str = "click_result",
) -> str:
    """Build a prompt for selecting which specific result to click."""
    instruction = "You decided to click on a search result. Which result would you click?"

    prompt = f"""{persona_desc}

Current search query: "{current_query}"

{memory_context}

Search results page:
{serp_description}

{instruction}
Respond with ONLY the number of the result you would click.

Answer:"""
    return prompt


def build_satisfaction_prompt(
    persona_desc: str,
    clicked_title: str,
    clicked_snippet: str,
    is_ad: bool,
) -> str:
    """Build a prompt to assess satisfaction after clicking a result."""
    result_type = "advertisement" if is_ad else "search result"
    prompt = f"""{persona_desc}

You clicked on the following {result_type}:
Title: {clicked_title}
Content: {clicked_snippet}

On a scale of 1-5, how satisfied are you with this result?
1 = Not at all relevant
2 = Slightly relevant
3 = Moderately relevant
4 = Very relevant
5 = Exactly what I needed

Respond with ONLY the number.

Rating:"""
    return prompt
