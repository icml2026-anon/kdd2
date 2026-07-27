"""Run search user simulation with LLM agents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from searchagent.config import SearchAgentConfig
from searchagent.evaluation.metrics import composite_fidelity_score
from searchagent.simulation.session_sim import SessionSimulator
from searchagent.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run SearchAgent simulation")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--num_sessions", type=int, default=None,
        help="Number of sessions to simulate (overrides config)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for results JSON",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    config = SearchAgentConfig.from_yaml(args.config)

    if args.num_sessions:
        config.simulation.num_sessions = args.num_sessions

    output_path = args.output or str(
        Path(config.simulation.output_dir) / "simulation_results.json"
    )

    simulator = SessionSimulator(config)
    results = simulator.run()

    # Save results
    simulator.save_results(output_path)

    # Evaluate
    metrics = composite_fidelity_score(results)
    logger.info("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

    # Save metrics
    metrics_path = Path(output_path).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
