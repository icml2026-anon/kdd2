"""Training data builder for SearchAgent++ calibration.

Parses real TREC sessions into supervised examples and caches LLM priors so the
calibrator can be trained offline (Phase A) without repeated API calls.

Per real interaction step we emit:
  - an ACTION example: (state features, gate features, log p_LLM(a), label a*)
  - CLICK examples:     (per-result exam/attr features, log p_LLM(j), clicked rank)
We also emit per-session moment features Phi(tau) for the session-level loss.

The action label for each step is derived from the real log:
  - first interaction          -> the user issued the initial query (no action label;
                                  it seeds state)
  - a click in an interaction  -> "click_result"
  - a later interaction's query that repeats topic words -> "reformulate"
                                  else "new_query"
  - last interaction with clicks -> session ends "stop_satisfied"
  - last interaction w/o clicks  -> session ends "stop_unsatisfied"
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from searchagent.agent.calibration import (
    ACTION_INDEX, CALIB_ACTIONS, N_ACTIONS, StateFeaturizer,
)
from searchagent.agent.persona import TRECPersonaExtractor
from searchagent.agent.prompts import (
    ACTION_LABELS, MACRO_ACTIONS, build_action_selection_prompt,
    build_click_selection_prompt,
)

logger = logging.getLogger(__name__)


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class PriorCache:
    """Disk-backed cache of LLM action/click priors keyed by prompt hash."""

    def __init__(self, path: str):
        self.path = path
        self.store: Dict[str, List[float]] = {}
        if os.path.exists(path):
            with open(path) as f:
                self.store = json.load(f)
            logger.info("Loaded %d cached priors from %s", len(self.store), path)

    def get(self, prompt: str, labels: List[str]) -> Optional[np.ndarray]:
        key = _hash(prompt + "||" + ",".join(labels))
        v = self.store.get(key)
        return np.array(v, dtype=np.float32) if v is not None else None

    def put(self, prompt: str, labels: List[str], probs: np.ndarray) -> None:
        key = _hash(prompt + "||" + ",".join(labels))
        self.store[key] = [float(x) for x in probs]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.store, f)
        logger.info("Saved %d priors to %s", len(self.store), self.path)


@dataclass
class ActionExample:
    feats: np.ndarray
    gate_feats: np.ndarray
    log_p_llm: np.ndarray      # [N_ACTIONS]
    label: int
    dataset: str


@dataclass
class ClickExample:
    exam_feats: np.ndarray     # [n_results, exam_dim]
    attr_feats: np.ndarray     # [n_results, attr_dim]
    clicked_idx: int
    dataset: str


@dataclass
class SessionMoments:
    phi: np.ndarray            # behavior moment vector Phi(tau)
    dataset: str


# Moment feature names for Phi(tau) (Component 5 / L_mom).
MOMENT_NAMES = [
    "click_rate", "reform_rate", "abandon_rate", "session_len",
    "click_rank_mean", "ad_click_rate", "dwell_mean",
]


def _serp_quality(results) -> float:
    """Mean relevance proxy in [0,1] for the SERP."""
    if not results:
        return 0.5
    rels = [getattr(r, "relevance", 0.5) for r in results]
    return float(np.clip(np.mean(rels), 0.0, 1.0))


def _topic_overlap(q1: str, q2: str) -> float:
    w1, w2 = set(q1.lower().split()), set(q2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def _exam_attr_feats(results, query: str, llm_click_probs: np.ndarray,
                     step: int, focus: float, expertise_idx: float):
    """Per-result examination and attractiveness feature rows."""
    n = len(results)
    exam = np.zeros((n, 7), dtype=np.float32)
    attr = np.zeros((n, 7), dtype=np.float32)
    for i, r in enumerate(results):
        rank = getattr(r, "rank", i + 1)
        is_ad = 1.0 if getattr(r, "is_ad", False) else 0.0
        rel = getattr(r, "relevance", 0.5)
        lp = float(np.log(max(llm_click_probs[i], 1e-9))) if i < len(llm_click_probs) else -9.0
        # examination: depends on geometry / fatigue / focus, NOT content
        exam[i] = [rank / 10.0, np.log(rank + 1) / 3.0, min(step / 10.0, 1.0),
                   focus, expertise_idx, is_ad, 1.0]
        # attractiveness: depends on content / relevance / llm preference
        attr[i] = [lp / 9.0, rel, is_ad, _topic_overlap(query, getattr(r, "title", "")),
                   rank / 10.0, expertise_idx, 1.0]
    return exam, attr


def build_dataset(
    sessions: List,
    dataset_name: str,
    llm,
    prior_cache: PriorCache,
    max_sessions: Optional[int] = None,
    save_every: int = 50,
):
    """Walk real sessions, derive (action, click) labels, cache LLM priors.

    Returns (action_examples, click_examples, session_moments).
    `llm` may be None to run cache-only (skips steps with no cached prior).
    """
    feaurizer = StateFeaturizer(n_intent=3)
    actions: List[ActionExample] = []
    clicks: List[ClickExample] = []
    moments: List[SessionMoments] = []

    sess = sessions[:max_sessions] if max_sessions else sessions
    # Dataset-aware stats (normalized): [clicks/session, abandon, reform].
    # Lets a single shared calibrator adapt its policy per dataset.
    from searchagent.evaluation.reference_data import compute_reference_from_sessions
    _ref = compute_reference_from_sessions(sess)
    ds_stats = [min(_ref.clicks_per_session / 5.0, 1.0),
                float(_ref.abandon_rate), float(_ref.reformulation_rate)]
    n_api = 0
    for si, s in enumerate(sess):
        persona = TRECPersonaExtractor.create_persona(s, persona_id=si)
        persona_desc = persona.to_prompt_description() if hasattr(persona, "to_prompt_description") else str(persona)
        expertise_idx = (["novice", "intermediate", "expert"].index(
            getattr(persona, "search_expertise", "intermediate")) / 2.0)

        # running cognitive proxies (numeric, mirrors CognitiveSearchState init)
        cog = type("C", (), dict(need_clarity=0.3, knowledge_state=0.0,
                                 frustration=0.0, exploration_focus=0.3))()
        n_click = n_skip = ad_clicked = 0
        last_satis = 0.0
        intent_dist = [0.6, 0.3, 0.1]

        inter = s.interactions
        for ti, it in enumerate(inter):
            results = it.results
            if not results:
                continue
            serp_desc = "\n".join(r.to_display() for r in results)
            mem_ctx = f"Step {ti}: {n_click} clicks, {n_skip} skips so far."

            # --- LLM action prior ---
            a_prompt = build_action_selection_prompt(
                persona_desc=persona_desc, current_query=it.query,
                serp_description=serp_desc, memory_context=mem_ctx,
                intent_description="exploratory", css_context="",
            )
            a_prior = prior_cache.get(a_prompt, ACTION_LABELS)
            if a_prior is None and llm is not None:
                pr = llm.get_action_logprobs(a_prompt, ACTION_LABELS)
                a_prior = np.array([pr.get(l, 1e-8) for l in ACTION_LABELS], dtype=np.float32)
                prior_cache.put(a_prompt, ACTION_LABELS, a_prior)
                n_api += 1
            if a_prior is None:
                continue
            # map the 7 ACTION_LABELS (MACRO_ACTIONS order) to CALIB_ACTIONS order
            a_prior_calib = np.array(
                [a_prior[MACRO_ACTIONS.index(act)] if act in MACRO_ACTIONS else 1e-8
                 for act in CALIB_ACTIONS], dtype=np.float32)
            a_prior_calib /= a_prior_calib.sum()
            log_p = np.log(np.clip(a_prior_calib, 1e-9, 1.0))

            # --- state + gate features ---
            mem_feats = dict(n_click=min(n_click / 5.0, 1.0), n_skip=min(n_skip / 5.0, 1.0),
                             ad_clicked=float(ad_clicked > 0), last_satisfaction=last_satis / 5.0,
                             page=1.0, step=min(ti / 10.0, 1.0))
            sq = _serp_quality(results)
            sf = feaurizer.build(persona, cog, intent_dist, mem_feats, sq, a_prior_calib)
            ent = sf.llm_entropy
            gate = np.array([ent, min(ti / 10.0, 1.0), min(n_click / 5.0, 1.0),
                             cog.frustration, cog.knowledge_state, sq], dtype=np.float32)

            # --- derive real action label for this step ---
            has_click = len(it.clicks) > 0
            is_last = (ti == len(inter) - 1)
            if has_click:
                label = "click_result"
            elif is_last:
                label = "stop_satisfied" if any(len(x.clicks) for x in inter) else "stop_unsatisfied"
            else:
                nxt = inter[ti + 1].query if ti + 1 < len(inter) else ""
                label = "reformulate" if _topic_overlap(it.query, nxt) >= 0.3 else "new_query"
            actions.append(ActionExample(sf.vector(), gate, log_p, ACTION_INDEX[label], dataset_name))

            # --- click example (when there is a real click) ---
            if has_click:
                clicked_rank = it.clicks[0].rank
                idxs = [i for i, r in enumerate(results) if getattr(r, "rank", i + 1) == clicked_rank]
                if idxs:
                    c_prompt = build_click_selection_prompt(
                        persona_desc=persona_desc, current_query=it.query,
                        serp_description=serp_desc, memory_context=mem_ctx,
                        click_type="click_result")
                    rank_labels = [str(getattr(r, "rank", i + 1)) for i, r in enumerate(results)]
                    c_prior = prior_cache.get(c_prompt, rank_labels)
                    if c_prior is None and llm is not None:
                        pr = llm.get_action_logprobs(c_prompt, rank_labels)
                        c_prior = np.array([pr.get(l, 1e-8) for l in rank_labels], dtype=np.float32)
                        prior_cache.put(c_prompt, rank_labels, c_prior)
                        n_api += 1
                    if c_prior is not None:
                        ef, af = _exam_attr_feats(results, it.query, c_prior, ti,
                                                  cog.exploration_focus, expertise_idx)
                        clicks.append(ClickExample(ef, af, idxs[0], dataset_name))

                # update running proxies on click
                n_click += 1
                if any(getattr(results[i], "is_ad", False) for i in idxs):
                    ad_clicked += 1
                last_satis = 4.0
                cog.knowledge_state = min(1.0, cog.knowledge_state + 0.1)
            else:
                n_skip += 1
                cog.frustration = min(1.0, cog.frustration + 0.1)

        # --- session moments Phi(tau) ---
        n_inter = max(len(inter), 1)
        tot_click = s.total_clicks
        ranks = s.click_positions or [0]
        ad_clicks = sum(1 for it in inter for c in it.clicks
                        for r in it.results if getattr(r, "rank", -1) == c.rank and getattr(r, "is_ad", False))
        dwell = [c.dwell_time for it in inter for c in it.clicks]
        phi = np.array([
            tot_click / n_inter,                       # click_rate
            float(s.num_queries > 1),                  # reform_rate (per session)
            float(tot_click == 0),                     # abandon_rate
            min(n_inter / 10.0, 1.0),                  # session_len
            float(np.mean(ranks)) / 10.0,              # click_rank_mean
            ad_clicks / max(tot_click, 1),             # ad_click_rate
            float(np.mean(dwell)) / 100.0 if dwell else 0.0,  # dwell_mean
        ], dtype=np.float32)
        moments.append(SessionMoments(phi, dataset_name))

        if llm is not None and (si + 1) % save_every == 0:
            prior_cache.save()
            logger.info("  built %d/%d sessions (%d API calls so far)", si + 1, len(sess), n_api)

    if llm is not None:
        prior_cache.save()
    logger.info("Dataset %s: %d action ex, %d click ex, %d sessions, %d new API calls",
                dataset_name, len(actions), len(clicks), len(moments), n_api)
    return actions, clicks, moments
