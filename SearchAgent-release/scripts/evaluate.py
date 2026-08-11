"""Evaluate simulated sessions against behavioral fidelity metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from searchagent.environment.search_env import SessionRecord
from searchagent.evaluation.metrics import composite_fidelity_score
from searchagent.utils.logging import setup_logging


def load_sessions(path: str) -> list:
    with open(path) as f:
        data = json.load(f)

    records = []
    for d in data:
        from searchagent.environment.search_env import ClickRecord
        clicks = []
        saved_clicks = d.get("clicks", [])
        if saved_clicks:
            for click in saved_clicks:
                clicks.append(ClickRecord(
                    step=click.get("step", 0),
                    query=click.get("query", ""),
                    rank=click.get("rank", 1),
                    title=click.get("title", ""),
                    is_ad=click.get("is_ad", False),
                    relevance=click.get("relevance", 0.5),
                    satisfaction=click.get("satisfaction", 0.0),
                    dwell_time=click.get("dwell_time", 0.0),
                ))
        else:
            for step in d.get("steps", []):
                if step.get("action", "").startswith("click") and "rank" in step:
                    clicks.append(ClickRecord(
                        step=step.get("step", 0),
                        query=step.get("query", ""),
                        rank=step.get("rank", 1),
                        title=step.get("title", ""),
                        is_ad=step.get("is_ad", False),
                        relevance=step.get("relevance", 0.5),
                        satisfaction=step.get("satisfaction", 0.0),
                        dwell_time=step.get("dwell_time", 0.0),
                    ))

        record = SessionRecord(
            persona_id=d.get("persona_id", 0),
            initial_query=d.get("initial_query", ""),
            steps=d.get("steps", []),
            clicks=clicks,
            queries=d.get("queries", []),
            total_steps=d.get("total_steps", 0),
            abandoned=d.get("abandoned", False),
        )
        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate simulation results")
    parser.add_argument("input", type=str, help="Path to simulation results JSON")
    parser.add_argument("--output", type=str, default=None, help="Output metrics JSON")
    args = parser.parse_args()

    logger = setup_logging("INFO")
    sessions = load_sessions(args.input)
    logger.info("Loaded %d sessions from %s", len(sessions), args.input)

    metrics = composite_fidelity_score(sessions)
    logger.info("=== Behavioral Fidelity Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
