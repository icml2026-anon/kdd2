from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class LLMConfig:
    model_name: str = "deepseek-v4-pro"
    temperature: float = 0.0
    max_new_tokens: int = 256
    top_p: float = 1.0
    reasoning_effort: str = "medium"


@dataclass
class PersonaConfig:
    age_range: List[int] = field(default_factory=lambda: [18, 65])
    search_expertise: List[str] = field(default_factory=lambda: ["novice", "intermediate", "expert"])
    ad_sensitivity: List[str] = field(default_factory=lambda: ["ad_tolerant", "ad_neutral", "ad_averse"])
    decision_style: List[str] = field(default_factory=lambda: ["impulsive", "deliberate", "price_sensitive"])
    patience_level: List[str] = field(default_factory=lambda: ["low", "medium", "high"])


@dataclass
class EnvironmentConfig:
    max_results_per_page: int = 10
    max_serp_pages: int = 3
    max_ads_per_page: int = 3
    ad_positions: List[int] = field(default_factory=lambda: [0, 3, 7])
    position_bias_click_prob: List[float] = field(default_factory=lambda: [0.30, 0.18, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01])


@dataclass
class SimulationConfig:
    max_session_steps: int = 20
    max_queries_per_session: int = 20
    num_sessions: int = 100
    seed: int = 42
    log_dir: str = "./logs"
    output_dir: str = "./output"


@dataclass
class CalibrationConfig:
    calibrator_path: Optional[str] = None
    hidden1: int = 64
    hidden2: int = 32
    dropout: float = 0.1
    alpha_max: float = 2.0
    exam_dim: int = 5
    attr_dim: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    action_batch_size: int = 512
    click_batch_size: int = 256
    grad_clip: float = 1.0
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4
    epsilon: float = 1e-8

    @property
    def hidden(self) -> int:
        return self.hidden1


@dataclass
class IntentConfig:
    states: List[str] = field(default_factory=lambda: ["exploratory", "focused", "transactional"])
    initial_distribution: List[float] = field(default_factory=lambda: [0.6, 0.3, 0.1])
    momentum: float = 0.8
    transitions: Dict[str, List[float]] = field(default_factory=lambda: {
        "click_result": [0.20, 0.60, 0.20],
        "click_organic": [0.20, 0.60, 0.20],
        "click_ad": [0.20, 0.60, 0.20],
        "skip": [0.45, 0.45, 0.10],
        "next_page": [0.55, 0.35, 0.10],
        "reformulate": [0.25, 0.65, 0.10],
        "new_query": [0.65, 0.25, 0.10],
        "stop_satisfied": [0.10, 0.75, 0.15],
        "stop_unsatisfied": [0.40, 0.45, 0.15],
    })


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "click_position_jsd",
        "session_length_jsd",
        "clicks_per_session_gap",
        "abandon_rate_gap",
        "reformulation_rate_gap",
    ])
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
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "SearchAgentConfig":
        llm_raw = {k: v for k, v in d.get("llm", {}).items() if k in LLMConfig.__dataclass_fields__}
        llm = LLMConfig(**llm_raw)
        persona_raw = {k: v for k, v in d.get("persona", {}).items() if k in PersonaConfig.__dataclass_fields__}
        persona = PersonaConfig(**persona_raw)
        env_raw = dict(d.get("environment", {}) or {})
        pos_bias = env_raw.pop("position_bias", {}) or {}
        if pos_bias:
            env_raw["position_bias_click_prob"] = pos_bias.get("click_prob_by_rank", EnvironmentConfig().position_bias_click_prob)
        env_raw = {k: v for k, v in env_raw.items() if k in EnvironmentConfig.__dataclass_fields__}
        env = EnvironmentConfig(**env_raw)
        sim_raw = {k: v for k, v in d.get("simulation", {}).items() if k in SimulationConfig.__dataclass_fields__}
        sim = SimulationConfig(**sim_raw)
        intent_raw = d.get("intent", {}) or {}
        intent = IntentConfig(
            states=intent_raw.get("states", IntentConfig().states),
            initial_distribution=intent_raw.get("initial_distribution", IntentConfig().initial_distribution),
            momentum=float(intent_raw.get("momentum", IntentConfig().momentum)),
            transitions=intent_raw.get("transitions", IntentConfig().transitions),
        )
        eval_raw = d.get("evaluation", {}) or {}
        evaluation = EvaluationConfig(
            metrics=eval_raw.get("metrics", EvaluationConfig().metrics),
            cfs_weights=eval_raw.get("cfs_weights", {}),
        )
        calib_raw = dict(d.get("calibration", {}) or {})
        if "hidden" in calib_raw and "hidden1" not in calib_raw:
            calib_raw["hidden1"] = calib_raw["hidden"]
        calib_raw = {k: v for k, v in calib_raw.items() if k in CalibrationConfig.__dataclass_fields__}
        calibration = CalibrationConfig(**calib_raw)
        return cls(llm=llm, persona=persona, environment=env, simulation=sim, intent=intent, evaluation=evaluation, calibration=calibration, device=d.get("device", "auto"))
