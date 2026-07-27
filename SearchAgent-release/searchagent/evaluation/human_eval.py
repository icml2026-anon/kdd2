"""Human evaluation framework for SearchAgent.

Implements a Turing-test-style discrimination task:
    Given a search session transcript, can a human judge tell whether it
    was produced by a real user or by SearchAgent?

Design follows best practices from:
    - Chiang & Lee, "Can Large Language Models Be an Alternative to Human
      Evaluations?", ACL 2023.
    - Clark et al., "All That's 'Human' Is Not Gold: Evaluating Human
      Evaluation of Generated Text", ACL 2021.

Protocol:
    1. Sample N real sessions from TREC Session Track 2014.
    2. Sample N simulated sessions from SearchAgent (matched by topic).
    3. Shuffle and anonymize all 2N sessions.
    4. Present each session to K judges (inter-annotator agreement via Fleiss' κ).
    5. Each judge labels: "real" or "simulated" + confidence (1-5).
    6. Compute discrimination accuracy, agreement, and calibration.

A good simulator should yield judge accuracy close to 50% (random chance),
indicating that simulated sessions are indistinguishable from real ones.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EvalSession:
    """A session prepared for human evaluation."""
    session_id: str
    source: str  # "real" or "simulated"
    topic: str
    queries: List[str]
    clicks: List[Dict[str, Any]]  # [{rank, title, dwell_time}, ...]
    num_queries: int
    num_clicks: int
    abandoned: bool


@dataclass
class JudgeAnnotation:
    """A single judge's annotation for one session."""
    session_id: str
    judge_id: str
    predicted_source: str  # "real" or "simulated"
    confidence: int  # 1 (guess) to 5 (certain)
    reasoning: str = ""


@dataclass
class HumanEvalResults:
    """Aggregated results from human evaluation."""
    n_sessions: int
    n_judges: int
    n_annotations: int

    # Core metrics
    discrimination_accuracy: float  # overall accuracy of judges
    accuracy_on_real: float  # accuracy on real sessions
    accuracy_on_simulated: float  # accuracy on simulated sessions

    # Per-judge agreement
    fleiss_kappa: float  # inter-annotator agreement

    # Confidence calibration
    mean_confidence_correct: float
    mean_confidence_incorrect: float

    # Ideal: accuracy ≈ 0.5 (indistinguishable)
    # Ideal: kappa moderate (judges agree but can't tell)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_sessions": self.n_sessions,
            "n_judges": self.n_judges,
            "n_annotations": self.n_annotations,
            "discrimination_accuracy": round(self.discrimination_accuracy, 4),
            "accuracy_on_real": round(self.accuracy_on_real, 4),
            "accuracy_on_simulated": round(self.accuracy_on_simulated, 4),
            "fleiss_kappa": round(self.fleiss_kappa, 4),
            "mean_confidence_correct": round(self.mean_confidence_correct, 2),
            "mean_confidence_incorrect": round(self.mean_confidence_incorrect, 2),
        }


# ── Session preparation ──────────────────────────────────────────────────

def prepare_real_sessions(
    trec_xml_path: str,
    n_sessions: int = 50,
    seed: int = 42,
) -> List[EvalSession]:
    """Sample and anonymize real sessions from TREC data."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(trec_xml_path)
    root = tree.getroot()

    all_sessions = []
    for session_elem in root.findall(".//session"):
        sid = session_elem.get("num", "0")
        topic_elem = session_elem.find("topic")
        topic = topic_elem.text.strip() if topic_elem is not None and topic_elem.text else ""

        queries = []
        clicks = []
        for interaction in session_elem.findall("interaction"):
            query_elem = interaction.find("query")
            if query_elem is not None and query_elem.text:
                queries.append(query_elem.text.strip())

            for click in interaction.findall(".//click"):
                rank_elem = click.find("rank")
                rank = int(rank_elem.text) if rank_elem is not None and rank_elem.text else 0

                # Try to get snippet or title
                title = ""
                for tag in ["title", "snippet", "clueweb12id"]:
                    elem = click.find(tag)
                    if elem is not None and elem.text:
                        title = elem.text.strip()
                        break

                starttime = click.find("starttime")
                endtime = click.find("endtime")
                dwell = 0.0
                if starttime is not None and endtime is not None:
                    try:
                        dwell = float(endtime.text) - float(starttime.text)
                    except (ValueError, TypeError):
                        pass

                clicks.append({"rank": rank, "title": title, "dwell_time": round(dwell, 1)})

        if queries:
            all_sessions.append(EvalSession(
                session_id=f"real_{sid}",
                source="real",
                topic=topic,
                queries=queries,
                clicks=clicks,
                num_queries=len(queries),
                num_clicks=len(clicks),
                abandoned=len(clicks) == 0,
            ))

    rng = random.Random(seed)
    rng.shuffle(all_sessions)
    return all_sessions[:n_sessions]


def prepare_simulated_sessions(
    sessions_json_path: str,
    n_sessions: int = 50,
    seed: int = 42,
) -> List[EvalSession]:
    """Load and anonymize simulated sessions from SearchAgent output."""
    with open(sessions_json_path) as f:
        data = json.load(f)

    all_sessions = []
    for i, d in enumerate(data):
        queries = d.get("queries", [d.get("initial_query", "")])
        clicks = []
        for step in d.get("steps", []):
            if step.get("action") in ("click_organic", "click_ad"):
                clicks.append({
                    "rank": step.get("rank", 0),
                    "title": step.get("title", ""),
                    "dwell_time": round(step.get("dwell_time", 0.0), 1),
                })

        all_sessions.append(EvalSession(
            session_id=f"sim_{i}",
            source="simulated",
            topic=d.get("initial_query", ""),
            queries=queries if isinstance(queries, list) else [queries],
            clicks=clicks,
            num_queries=len(queries) if isinstance(queries, list) else 1,
            num_clicks=len(clicks),
            abandoned=d.get("abandoned", False),
        ))

    rng = random.Random(seed)
    rng.shuffle(all_sessions)
    return all_sessions[:n_sessions]


def create_evaluation_set(
    real_sessions: List[EvalSession],
    simulated_sessions: List[EvalSession],
    seed: int = 42,
) -> Tuple[List[EvalSession], Dict[str, str]]:
    """Merge, shuffle, and re-ID sessions for blind evaluation.

    Returns:
        (shuffled_sessions, answer_key: {new_id → "real"/"simulated"})
    """
    combined = []
    rng = random.Random(seed)

    for s in real_sessions + simulated_sessions:
        combined.append(s)

    rng.shuffle(combined)

    answer_key = {}
    blinded = []
    for idx, s in enumerate(combined):
        new_id = f"S{idx + 1:03d}"
        answer_key[new_id] = s.source
        blinded.append(EvalSession(
            session_id=new_id,
            source="hidden",  # blinded
            topic=s.topic,
            queries=s.queries,
            clicks=s.clicks,
            num_queries=s.num_queries,
            num_clicks=s.num_clicks,
            abandoned=s.abandoned,
        ))

    return blinded, answer_key


def format_session_for_display(session: EvalSession) -> str:
    """Format a session as a human-readable transcript for evaluation."""
    lines = [f"Session ID: {session.session_id}"]
    lines.append(f"Topic: {session.topic}")
    lines.append(f"Number of queries: {session.num_queries}")
    lines.append(f"Number of clicks: {session.num_clicks}")
    lines.append(f"Abandoned: {'Yes' if session.abandoned else 'No'}")
    lines.append("")

    for i, q in enumerate(session.queries, 1):
        lines.append(f"  Query {i}: {q}")

    if session.clicks:
        lines.append("")
        lines.append("  Clicks:")
        for j, c in enumerate(session.clicks, 1):
            dwell = f", dwell={c['dwell_time']}s" if c.get("dwell_time") else ""
            lines.append(f"    {j}. Rank {c['rank']}: {c.get('title', '(no title)')}{dwell}")
    else:
        lines.append("  (No clicks)")

    return "\n".join(lines)


# ── Evaluation scoring ────────────────────────────────────────────────────

def compute_human_eval_results(
    annotations: List[JudgeAnnotation],
    answer_key: Dict[str, str],
) -> HumanEvalResults:
    """Compute evaluation metrics from judge annotations.

    Args:
        annotations: All judge annotations.
        answer_key: {session_id → "real"/"simulated"} ground truth.
    """
    if not annotations:
        return HumanEvalResults(
            n_sessions=0, n_judges=0, n_annotations=0,
            discrimination_accuracy=0.5, accuracy_on_real=0.5,
            accuracy_on_simulated=0.5, fleiss_kappa=0.0,
            mean_confidence_correct=0.0, mean_confidence_incorrect=0.0,
        )

    judges = set(a.judge_id for a in annotations)
    sessions = set(a.session_id for a in annotations)

    correct = 0
    total = 0
    correct_real = 0
    total_real = 0
    correct_sim = 0
    total_sim = 0
    conf_correct = []
    conf_incorrect = []

    for a in annotations:
        true_source = answer_key.get(a.session_id)
        if true_source is None:
            continue

        is_correct = a.predicted_source == true_source
        total += 1
        if is_correct:
            correct += 1
            conf_correct.append(a.confidence)
        else:
            conf_incorrect.append(a.confidence)

        if true_source == "real":
            total_real += 1
            if is_correct:
                correct_real += 1
        else:
            total_sim += 1
            if is_correct:
                correct_sim += 1

    accuracy = correct / total if total > 0 else 0.5
    acc_real = correct_real / total_real if total_real > 0 else 0.5
    acc_sim = correct_sim / total_sim if total_sim > 0 else 0.5

    # Fleiss' kappa
    kappa = _fleiss_kappa(annotations, answer_key)

    return HumanEvalResults(
        n_sessions=len(sessions),
        n_judges=len(judges),
        n_annotations=total,
        discrimination_accuracy=accuracy,
        accuracy_on_real=acc_real,
        accuracy_on_simulated=acc_sim,
        fleiss_kappa=kappa,
        mean_confidence_correct=float(np.mean(conf_correct)) if conf_correct else 0.0,
        mean_confidence_incorrect=float(np.mean(conf_incorrect)) if conf_incorrect else 0.0,
    )


def _fleiss_kappa(
    annotations: List[JudgeAnnotation],
    answer_key: Dict[str, str],
) -> float:
    """Compute Fleiss' kappa for inter-annotator agreement.

    Categories: "real", "simulated".
    """
    # Group annotations by session
    session_annotations: Dict[str, List[str]] = {}
    for a in annotations:
        if a.session_id not in session_annotations:
            session_annotations[a.session_id] = []
        session_annotations[a.session_id].append(a.predicted_source)

    if not session_annotations:
        return 0.0

    categories = ["real", "simulated"]
    n_categories = len(categories)
    n_subjects = len(session_annotations)

    # All subjects must have same number of raters
    n_raters_list = [len(v) for v in session_annotations.values()]
    if len(set(n_raters_list)) != 1:
        # Unequal raters — use minimum
        n_raters = min(n_raters_list)
    else:
        n_raters = n_raters_list[0]

    if n_raters < 2:
        return 0.0

    # Build count matrix: n_subjects × n_categories
    counts = np.zeros((n_subjects, n_categories))
    for i, (sid, labels) in enumerate(session_annotations.items()):
        for label in labels[:n_raters]:
            if label in categories:
                counts[i, categories.index(label)] += 1

    # Fleiss' kappa computation
    N = n_subjects
    n = n_raters
    k = n_categories

    # p_j = proportion of all assignments to category j
    p_j = counts.sum(axis=0) / (N * n)

    # P_i = extent of agreement for each subject
    P_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1)) if n > 1 else np.zeros(N)

    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0

    kappa = (P_bar - P_e) / (1.0 - P_e)
    return float(kappa)


# ── Export utilities ──────────────────────────────────────────────────────

def export_evaluation_set(
    sessions: List[EvalSession],
    answer_key: Dict[str, str],
    output_dir: str,
) -> None:
    """Export evaluation materials to files.

    Creates:
        - eval_sessions.json: Blinded session data for judges
        - eval_transcripts.txt: Human-readable session transcripts
        - answer_key.json: Ground truth (for scoring only)
        - eval_instructions.md: Instructions for judges
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Session data (machine-readable)
    session_data = []
    for s in sessions:
        session_data.append({
            "session_id": s.session_id,
            "topic": s.topic,
            "queries": s.queries,
            "clicks": s.clicks,
            "num_queries": s.num_queries,
            "num_clicks": s.num_clicks,
            "abandoned": s.abandoned,
        })
    with open(out / "eval_sessions.json", "w") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    # Transcripts (human-readable)
    with open(out / "eval_transcripts.txt", "w") as f:
        for s in sessions:
            f.write(format_session_for_display(s))
            f.write("\n" + "=" * 60 + "\n\n")

    # Answer key (keep secret)
    with open(out / "answer_key.json", "w") as f:
        json.dump(answer_key, f, indent=2)

    # Instructions
    instructions = _generate_instructions(len(sessions))
    with open(out / "eval_instructions.md", "w") as f:
        f.write(instructions)


def _generate_instructions(n_sessions: int) -> str:
    """Generate evaluation instructions for judges."""
    return f"""# Search Session Discrimination Task

## Task Description

You will be shown {n_sessions} search session transcripts. Each session records
a user's search behavior, including:

- The search queries they issued
- Which results they clicked (by rank position)
- How long they spent on each result (dwell time)
- Whether they abandoned the search

**Your task**: For each session, determine whether it was produced by a
**real human user** or by an **AI-simulated agent**.

## Instructions

1. Read each session transcript carefully.
2. For each session, provide:
   - **Label**: "real" or "simulated"
   - **Confidence**: 1 (pure guess) to 5 (very certain)
   - **Reasoning**: Brief explanation of your decision (optional but appreciated)

3. Consider behavioral cues such as:
   - Query formulation patterns (natural vs. formulaic)
   - Click position patterns (position bias)
   - Dwell time patterns (too uniform?)
   - Session length and flow (natural progression?)
   - Reformulation strategies (meaningful refinement?)

## Important Notes

- Exactly half of the sessions are real and half are simulated.
- There are no tricks — focus on behavioral plausibility.
- Work through sessions in order; do not go back to change answers.
- This task typically takes 15–20 minutes.

## Output Format

Please record your answers in the following JSON format:

```json
[
  {{"session_id": "S001", "label": "real", "confidence": 3, "reasoning": "..."}},
  {{"session_id": "S002", "label": "simulated", "confidence": 4, "reasoning": "..."}},
  ...
]
```

Thank you for participating!
"""


def format_results_latex(results: HumanEvalResults) -> str:
    """Format human evaluation results as a LaTeX table."""
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Human evaluation: discrimination task results}",
        "\\label{tab:human_eval}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
        f"Sessions evaluated & {results.n_sessions} \\\\",
        f"Number of judges & {results.n_judges} \\\\",
        f"Overall accuracy & {results.discrimination_accuracy:.1%} \\\\",
        f"Accuracy on real & {results.accuracy_on_real:.1%} \\\\",
        f"Accuracy on simulated & {results.accuracy_on_simulated:.1%} \\\\",
        f"Fleiss' $\\kappa$ & {results.fleiss_kappa:.3f} \\\\",
        f"Confidence (correct) & {results.mean_confidence_correct:.1f}/5 \\\\",
        f"Confidence (incorrect) & {results.mean_confidence_incorrect:.1f}/5 \\\\",
        "\\bottomrule",
        "\\multicolumn{2}{l}{\\footnotesize Ideal accuracy $\\approx$ 50\\% "
        "(indistinguishable from real).} \\\\",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)
