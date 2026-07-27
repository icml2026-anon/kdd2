"""Configuration management for SearchAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class LLMConfig:
    # SearchAgent is API-only. No local model is loaded; all generation and
    # forced-choice logprobs go through the DeepSeek chat-completions API.
    model_name: str = "deepseek-v4-flash"
    temperature: float = 0.7
    max_new_tokens: int = 256
    top_p: float = 0.9


@dataclass
class PersonaConfig:
    age_range: List[int] = field(default_factory=lambda: [18, 65])
    search_expertise: List[str] = field(
        default_factory=lambda: ["novice", "intermediate", "expert"]
    )
    ad_sensitivity: List[str] = field(
        default_factory=lambda: ["ad_tolerant", "ad_neutral", "ad_averse"]
    )
    decision_style: List[str] = field(
        default_factory=lambda: ["impulsive", "deliberate", "price_sensitive"]
    )
    patience_level: List[str] = field(
        default_factory=lambda: ["low", "medium", "high"]
    )


@dataclass
class EnvironmentConfig:
    max_results_per_page: int = 10
    max_ads_per_page: int = 3
    ad_positions: List[int] = field(default_factory=lambda: [0, 3, 7])
    position_bias_click_prob: List[float] = field(
        default_factory=lambda: [0.30, 0.18, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01]
    )


@dataclass
class SimulationConfig:
    max_session_steps: int = 20
    max_queries_per_session: int = 5
    session_timeout_prob: float = 0.05
    pre_click_abandon_prob: float = 0.0   # prob of zero-click abandon at step 0 (high-abandon datasets)
    num_sessions: int = 100
    seed: int = 42
    log_dir: str = "./logs"
    output_dir: str = "./output"


@dataclass
class CalibrationConfig:
    """SearchAgent++ learned-calibrator config (replaces v1 hand-tuned dynamics).

    All values that used to be hardcoded in calibration.py / train_calibrator.py
    live here so nothing behavioral is buried in code.
    """
    calibrator_path: Optional[str] = None  # path to trained calibrator.pt
    hidden: int = 64
    exam_dim: int = 7
    attr_dim: int = 7
    lambda_click: float = 1.0
    lambda_kl: float = 0.1
    lambda_cal: float = 0.05
    lr: float = 3e-3
    epochs: int = 400
    click_temperature: float = 1.0   # >1 softens click-position dist (inference-time, no retrain)
    target_clicks: float = 0.0       # soft cap on clicks/session (0 = disabled); curbs per-seed click blow-ups
    click_boost: float = 1.0         # >1 boosts click prob while below target (for under-clicking datasets)


@dataclass
class IntentConfig:
    states: List[str] = field(
        default_factory=lambda: ["exploratory", "focused", "transactional"]
    )
    initial_distribution: List[float] = field(
        default_factory=lambda: [0.6, 0.3, 0.1]
    )
    transitions: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(
        default_factory=lambda: [
            "click_position_jsd",
            "session_length_jsd",
            "clicks_per_session_gap",
            "abandon_rate_gap",
            "reformulation_rate_gap",
        ]
    )
    cfs_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class SearchAgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str) -> "SearchAgentConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "SearchAgentConfig":
        # Tolerate legacy/unknown keys (e.g. local-model fields removed in the
        # API-only refactor: model_path, torch_dtype, device_map).
        llm_raw = {k: v for k, v in d.get("llm", {}).items()
                   if k in LLMConfig.__dataclass_fields__}
        llm = LLMConfig(**llm_raw)
        persona = PersonaConfig(**d.get("persona", {}))

        env_raw = d.get("environment", {})
        pos_bias = env_raw.pop("position_bias", {})
        if pos_bias:
            env_raw["position_bias_click_prob"] = pos_bias.get(
                "click_prob_by_rank",
                EnvironmentConfig().position_bias_click_prob,
            )
        env = EnvironmentConfig(**env_raw)

        # Tolerate legacy v1 dynamics keys (fatigue_*/satiation_*/satisfaction_*)
        # that may still linger in older configs.
        sim_raw = {k: v for k, v in d.get("simulation", {}).items()
                   if k in SimulationConfig.__dataclass_fields__}
        sim = SimulationConfig(**sim_raw)

        intent_raw = d.get("intent", {})
        intent = IntentConfig(
            states=intent_raw.get("states", IntentConfig().states),
            initial_distribution=intent_raw.get(
                "initial_distribution", IntentConfig().initial_distribution
            ),
            transitions=intent_raw.get("transitions", {}),
        )

        eval_raw = d.get("evaluation", {})
        evaluation = EvaluationConfig(
            metrics=eval_raw.get("metrics", []),
            cfs_weights=eval_raw.get("cfs_weights", {}),
        )

        calib_raw = {k: v for k, v in d.get("calibration", {}).items()
                     if k in CalibrationConfig.__dataclass_fields__}
        calibration = CalibrationConfig(**calib_raw)

        return cls(
            llm=llm,
            persona=persona,
            environment=env,
            simulation=sim,
            intent=intent,
            evaluation=evaluation,
            calibration=calibration,
            device=d.get("device", "auto"),
        )
