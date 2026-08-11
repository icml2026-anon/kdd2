from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from searchagent.agent.calibration import ACTION_INDEX, CALIB_ACTIONS, StateFeaturizer
from searchagent.agent.memory import AgentMemory, MemoryEntry
from searchagent.agent.persona import TRECPersonaExtractor
from searchagent.agent.prompts import (
    ACTION_LABELS,
    MACRO_ACTIONS,
    QUALITY_LABELS,
    UTILITY_LABELS,
    build_action_selection_prompt,
    build_click_selection_prompt,
    build_satisfaction_prompt,
    build_serp_quality_prompt,
)
from searchagent.agent.search_agent import CognitiveSearchState, IntentTracker
from searchagent.config import IntentConfig


logger = logging.getLogger(__name__)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class PriorCache:
    def __init__(self, path: str):
        self.path = path
        self.store: Dict[str, List[float]] = {}
        if os.path.exists(path):
            with open(path) as f:
                self.store = json.load(f)

    def get(self, prompt: str, labels: List[str]) -> Optional[np.ndarray]:
        value = self.store.get(_hash(prompt + "||" + ",".join(labels)))
        return np.asarray(value, dtype=np.float32) if value is not None else None

    def put(self, prompt: str, labels: List[str], probs: np.ndarray) -> None:
        self.store[_hash(prompt + "||" + ",".join(labels))] = [float(x) for x in probs]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.store, f)


@dataclass
class ActionExample:
    feats: np.ndarray
    gate_feats: np.ndarray
    log_p_llm: np.ndarray
    label: int
    dataset: str


@dataclass
class ClickExample:
    exam_feats: np.ndarray
    attr_feats: np.ndarray
    ranks: np.ndarray
    clicked_idx: int
    dataset: str


@dataclass
class SessionMoments:
    phi: np.ndarray
    dataset: str


MOMENT_NAMES = ["click_rate", "reform_rate", "abandon_rate", "session_len", "click_rank_mean", "ad_click_rate", "dwell_mean"]


def _topic_overlap(q1: str, q2: str) -> float:
    a = set(q1.lower().split())
    b = set(q2.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cached_probs(cache: PriorCache, llm, prompt: str, labels: List[str], rotation_key: str) -> Optional[np.ndarray]:
    prior = cache.get(prompt, labels)
    if prior is not None:
        return prior
    if llm is None:
        return None
    values = llm.get_action_logprobs(prompt, labels, rotation_key=rotation_key)
    prior = np.asarray([values[label] for label in labels], dtype=np.float32)
    cache.put(prompt, labels, prior)
    return prior


def _visible_result(result) -> dict:
    title = str(getattr(result, "title", "") or "")
    snippet = str(getattr(result, "snippet", "") or "")
    url = str(getattr(result, "url", "") or "")
    masks = {
        "title": bool(getattr(result, "title_available", bool(title))),
        "snippet": bool(getattr(result, "snippet_available", bool(snippet))),
        "url": bool(getattr(result, "url_available", bool(url))),
    }
    item = {
        "title": title if masks["title"] else "",
        "snippet": snippet if masks["snippet"] else "",
        "url": url if masks["url"] else "",
        "field_masks": masks,
        "text_blind": not any(masks.values()),
    }
    opaque_document_id = str(getattr(result, "opaque_document_id", "") or "")
    opaque_domain_id = str(getattr(result, "opaque_domain_id", "") or "")
    if opaque_document_id:
        item["opaque_document_id"] = opaque_document_id
    if opaque_domain_id:
        item["opaque_domain_id"] = opaque_domain_id
    return item


def _rank_free_payload(results, key: str):
    order = list(range(len(results)))
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & 0xFFFFFFFF
    np.random.RandomState(seed).shuffle(order)
    objects = []
    labels = []
    for out_idx, original_idx in enumerate(order):
        label = f"C{out_idx}"
        item = _visible_result(results[original_idx])
        item["candidate_label"] = label
        objects.append(item)
        labels.append(label)
    return order, objects, labels


def _action_label(interactions, index: int) -> str:
    current = interactions[index]
    if current.clicks:
        return "click_result"
    if index == len(interactions) - 1:
        return "stop_satisfied" if any(it.clicks for it in interactions[:index]) else "stop_unsatisfied"
    next_query = interactions[index + 1].query
    return "reformulate" if _topic_overlap(current.query, next_query) >= 0.3 else "new_query"


def build_dataset(sessions: List, dataset_name: str, llm, prior_cache: PriorCache, max_sessions: Optional[int] = None, save_every: int = 50):
    featurizer = StateFeaturizer(n_intent=3)
    actions: List[ActionExample] = []
    clicks: List[ClickExample] = []
    moments: List[SessionMoments] = []
    selected = sessions[:max_sessions] if max_sessions else sessions
    for session_index, session in enumerate(selected):
        persona = TRECPersonaExtractor.create_persona(session, persona_id=session_index)
        persona_desc = persona.to_dict()
        cognitive = CognitiveSearchState()
        intent = IntentTracker(IntentConfig())
        memory = AgentMemory(max_entries=8)
        n_click = 0
        n_skip = 0
        last_utility = 0.0
        interactions = session.interactions
        for step, interaction in enumerate(interactions):
            results = interaction.results
            if not results:
                continue
            visible_serp = {
                "query": interaction.query,
                "page": 1,
                "results": [
                    {"rank": int(getattr(r, "rank", i + 1)), **_visible_result(r)} for i, r in enumerate(results)
                ],
                "text_blind": all(_visible_result(r)["text_blind"] for r in results),
            }
            quality_prompt = build_serp_quality_prompt(interaction.query, visible_serp)
            quality_prior = _cached_probs(prior_cache, llm, quality_prompt, QUALITY_LABELS, f"{dataset_name}:{session_index}:{step}:quality")
            if quality_prior is None:
                continue
            quality = float(sum((i / 10.0) * quality_prior[i] for i in range(11)))
            action_prompt = build_action_selection_prompt(
                persona_desc,
                interaction.query,
                visible_serp,
                memory.get_recent_records(),
                intent.distribution.tolist(),
                cognitive.to_dict(),
                {"step": step, "page": 1},
            )
            action_prior_raw = _cached_probs(prior_cache, llm, action_prompt, ACTION_LABELS, f"{dataset_name}:{session_index}:{step}:action")
            if action_prior_raw is None:
                continue
            macro = dict(zip(MACRO_ACTIONS, action_prior_raw))
            action_prior = np.asarray([macro.get(action, 1e-8) for action in CALIB_ACTIONS], dtype=np.float64)
            action_prior = np.clip(action_prior, 1e-8, None)
            action_prior /= action_prior.sum()
            step_norm = min(step / 19.0, 1.0)
            click_norm = min(n_click / 20.0, 1.0)
            skip_norm = min(n_skip / 20.0, 1.0)
            memory_feats = {
                "n_click": click_norm,
                "n_skip": skip_norm,
                "last_satisfaction": float(np.clip((last_utility - 1.0) / 4.0, 0.0, 1.0)) if last_utility > 0 else 0.0,
                "page": 0.0,
                "step": step_norm,
            }
            state = featurizer.build(persona, cognitive, intent.distribution.tolist(), memory_feats, quality, action_prior)
            gate = np.asarray([state.llm_entropy, quality, step_norm, click_norm, cognitive.frustration, cognitive.knowledge_state], dtype=np.float32)
            label = _action_label(interactions, step)
            actions.append(ActionExample(state.vector(), gate, np.log(action_prior).astype(np.float32), ACTION_INDEX[label], dataset_name))
            if interaction.clicks:
                clicked_rank = interaction.clicks[0].rank
                matching = [i for i, result in enumerate(results) if int(getattr(result, "rank", i + 1)) == clicked_rank]
                if matching:
                    order, candidate_objects, candidate_labels = _rank_free_payload(results, f"{dataset_name}:{session_index}:{step}")
                    if all(item["text_blind"] for item in candidate_objects):
                        semantic_out = np.full(len(results), 1.0 / len(results), dtype=np.float32)
                    else:
                        select_prompt = build_click_selection_prompt(persona_desc, interaction.query, candidate_objects)
                        select_prior = _cached_probs(prior_cache, llm, select_prompt, candidate_labels, f"{dataset_name}:{session_index}:{step}:select")
                        if select_prior is None:
                            select_prior = np.full(len(results), 1.0 / len(results), dtype=np.float32)
                        semantic_out = select_prior
                    semantic = np.zeros(len(results), dtype=np.float64)
                    mtext = np.zeros(len(results), dtype=np.float64)
                    for out_idx, original_idx in enumerate(order):
                        semantic[original_idx] = semantic_out[out_idx]
                        mtext[original_idx] = 0.0 if candidate_objects[out_idx]["text_blind"] else 1.0
                    semantic = np.clip(semantic, 1e-8, None)
                    semantic /= semantic.sum()
                    expertise = {"novice": 0.0, "intermediate": 0.5, "expert": 1.0}.get(persona.search_expertise, 0.5)
                    style = {"impulsive": 0.0, "deliberate": 1.0, "price_sensitive": 0.5}.get(persona.decision_style, 0.5)
                    exam = np.asarray([0.0, step_norm, cognitive.exploration_focus, expertise, click_norm], dtype=np.float32)
                    attr = np.stack([np.log(semantic + 1e-8), np.full(len(results), style), mtext], axis=1).astype(np.float32)
                    ranks = np.asarray([max(1, int(getattr(r, "rank", i + 1))) for i, r in enumerate(results)], dtype=np.float32)
                    clicks.append(ClickExample(exam, attr, ranks, matching[0], dataset_name))
                    clicked_result = results[matching[0]]
                    feedback_prompt = build_satisfaction_prompt(
                        persona_desc,
                        getattr(clicked_result, "title", ""),
                        getattr(clicked_result, "snippet", ""),
                        getattr(clicked_result, "is_ad", False),
                        current_query=interaction.query,
                        clicked_url=getattr(clicked_result, "url", ""),
                    )
                    utility_prior = _cached_probs(prior_cache, llm, feedback_prompt, UTILITY_LABELS, f"{dataset_name}:{session_index}:{step}:utility")
                    if utility_prior is not None:
                        last_utility = float(sum((i + 1) * utility_prior[i] for i in range(5)))
                    else:
                        last_utility = 3.0
                    n_click += 1
                    cognitive.update("click_result", last_utility)
                    intent.update("click_result")
                    memory.add(MemoryEntry(step, interaction.query, "click_result", target=getattr(clicked_result, "url", ""), result_title=getattr(clicked_result, "title", ""), is_ad=getattr(clicked_result, "is_ad", False), satisfaction=last_utility, dwell_time=float(interaction.clicks[0].dwell_time)))
                    continue
            if label == "skip":
                n_skip += 1
            cognitive.update(label, 0.0)
            intent.update(label)
            target = interactions[step + 1].query if label in ("reformulate", "new_query") and step + 1 < len(interactions) else ""
            memory.add(MemoryEntry(step, interaction.query, label, target=target))
        n_inter = max(len(interactions), 1)
        ranks = session.click_positions or [0]
        dwells = [click.dwell_time for interaction in interactions for click in interaction.clicks]
        phi = np.asarray([
            session.total_clicks / n_inter,
            float(session.num_queries > 1),
            float(session.total_clicks == 0),
            min(n_inter / 20.0, 1.0),
            float(np.mean(ranks)) / 10.0,
            0.0,
            float(np.mean(dwells)) / 120.0 if dwells else 0.0,
        ], dtype=np.float32)
        moments.append(SessionMoments(phi, dataset_name))
        if llm is not None and (session_index + 1) % save_every == 0:
            prior_cache.save()
    if llm is not None:
        prior_cache.save()
    return actions, clicks, moments
