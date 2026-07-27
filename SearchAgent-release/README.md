# SearchAgent

**LLM Agent-based User Simulation for Search Advertising**

SearchAgent uses LLM-powered agents with rich personas to simulate realistic user behavior in search advertising scenarios. Each agent has configurable traits (ad sensitivity, patience, decision style, search expertise) that drive its search interactions, including organic result clicks, ad clicks, query reformulation, and session abandonment.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                      Experiment Runner                            │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                   Session Simulator                          │ │
│  │                                                              │ │
│  │  ┌────────────────────┐    ┌──────────────────────────────┐  │ │
│  │  │   Search Agent      │    │   Search Environment         │  │ │
│  │  │                    │    │                              │  │ │
│  │  │  Persona ──────────┤    │  SERP Generator              │  │ │
│  │  │  Memory ───────────┤◄──►│  (10 scenarios, fuzzy match) │  │ │
│  │  │  Intent Tracker    │    │                              │  │ │
│  │  │  LLM Backbone ────┤    │  Session Recorder            │  │ │
│  │  │  (Qwen2-7B)       │    │  (clicks, satisfaction,      │  │ │
│  │  │  Satisfaction Rater│    │   dwell time)                │  │ │
│  │  └────────────────────┘    └──────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────┐  ┌───────────────────────────────────┐  │
│  │  Baseline Agents     │  │   Evaluation & Analysis           │  │
│  │  Random             │  │   5 core metrics + CFS              │  │
│  │  Rule-Based         │  │   Per-trait breakdowns            │  │
│  │  Naive LLM          │  │   LaTeX tables + Markdown report  │  │
│  ├─────────────────────┤  └───────────────────────────────────┘  │
│  │  Ablation Agents     │                                        │
│  │  A1: w/o Persona     │                                        │
│  │  A2: w/o Intent      │                                        │
│  │  A3: w/o Memory      │                                        │
│  │  A4: w/o Ad Model    │                                        │
│  │  A5: w/o Pos. Bias   │                                        │
│  └─────────────────────┘                                         │
└───────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Persona-driven behavior**: Each agent has configurable traits (ad sensitivity, patience, decision style) that systematically bias its actions
- **Intent evolution**: Soft distribution over exploratory/focused/transactional states, updated by agent actions
- **Ad interaction modeling**: First framework to explicitly model user-ad interactions in LLM simulation
- **Forced-choice logprobs**: Action selection via token logprobs extraction (not free-text parsing)
- **Hybrid click selection**: LLM logprob preference × position bias prior for diverse, realistic click targets
- **Satisfaction assessment**: LLM-rated post-click satisfaction (1–5 scale) with dwell time estimation
- **Rich SERP templates**: 10 scenario-specific template pools with fuzzy query matching for reformulated queries
- **Data-driven evaluation**: 5-metric CFS grounded in TREC Session Track 2014 (1,257 real sessions), cross-validated against Joachims 2005, Jansen 2006, Li 2009
- **Statistical significance testing**: Bootstrap CI, paired permutation test (Dror et al. ACL 2018), Cohen's d effect size
- **Baseline & ablation framework**: 3 baselines (Random, Rule-Based, Naive LLM) + 5 ablation variants
- **Per-trait analysis**: Automatic behavioral pattern breakdowns by persona traits
- **LaTeX table generation**: Publication-ready comparison, ablation, and descriptive statistics tables
- **Local LLM**: Runs entirely on local Qwen2-7B-Instruct, no API dependency

## Project Structure

```
SearchAgent/
├── searchagent/
│   ├── agent/
│   │   ├── persona.py          # Persona generation (10 scenarios, 4 trait dimensions)
│   │   ├── memory.py           # Session memory tracking
│   │   ├── llm_backbone.py     # Local LLM (Qwen2-7B) wrapper with logprobs
│   │   ├── prompts.py          # Prompt templates for all interactions
│   │   ├── search_agent.py     # Core agent: persona bias, intent, satisfaction
│   │   ├── baselines.py        # RandomAgent, RuleBasedAgent, NaiveLLMAgent
│   │   └── ablations.py        # AblationAgent (A1–A5 variants)
│   ├── environment/
│   │   ├── serp.py             # SERP generation with fuzzy query matching
│   │   ├── serp_templates.py   # 10 scenario-specific template pools
│   │   └── search_env.py       # Search environment + session recording
│   ├── simulation/
│   │   └── session_sim.py      # Session simulation loop with satisfaction
│   ├── evaluation/
│   │   ├── metrics.py          # 5 core behavioral fidelity metrics + CFS
│   │   ├── reference_data.py   # TREC 2014 reference distributions (1,257 sessions)
│   │   ├── statistical_tests.py # Bootstrap CI, permutation test, Wilcoxon, Cohen's d
│   │   ├── human_eval.py       # Turing-test discrimination task, Fleiss' κ
│   │   ├── downstream_eval.py  # PBM click model, cross-training validation
│   │   └── analysis.py         # LaTeX tables, Markdown reports, trait analysis
│   ├── utils/
│   │   └── logging.py          # Logging utilities
│   └── config.py               # Configuration management
├── configs/
│   └── default.yaml            # Default configuration
├── scripts/
│   ├── run_simulation.py       # Run single simulation
│   ├── run_experiment.py       # Experiment runner + auto significance testing
│   └── evaluate.py             # Evaluate simulation results
├── tests/
│   ├── test_smoke.py           # core smoke tests
│   └── test_extended.py        # extended tests (baselines, ablations, metrics, stat, human/downstream eval)
└── output/                     # (generated at runtime; not versioned)
```

## Quick Start

```bash
# Install
pip install -e .

# Set API credentials (OpenAI-compatible endpoint, e.g. DeepSeek).
# Copy .env.example and fill in, or export directly:
export DEEPSEEK_API_KEY=sk-your-key-here
export DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions   # optional; this is the default

# Run all tests (no API key needed; a dummy key is used)
python -m pytest tests/ -v

# Run single simulation
python scripts/run_simulation.py --config configs/default.yaml --num_sessions 10

# Run full experiment: baselines + ablations + full model
python scripts/run_experiment.py --config configs/default.yaml \
    --agents full random rule_based naive_llm A1_no_persona A2_no_intent A3_no_memory A4_no_ad A5_no_posbias \
    --num_sessions 20 --seeds 42 123 456 \
    --output_dir output/experiments

# Evaluate results
python scripts/evaluate.py --input output/experiments/experiment_summary.json
```

> **Model backbone:** the agent selects actions via forced-choice token logprobs from an
> OpenAI-compatible chat API (default: DeepSeek `deepseek-v4-flash`). The endpoint must return
> `logprobs`. Configure the model name in `configs/*.yaml` under `llm.model_name`.

## Experiment Results

### Agent Comparison

CFS is bounded in [0, 1] where **lower = more realistic**. All metrics are grounded in real search log distributions (TREC Session Track 2014). Results below are from 20 sessions per agent.

| Agent | CFS ↓ | Click JSD ↓ | Session JSD ↓ | Clicks/S Δ ↓ | Abandon Δ ↓ | Reform. Δ ↓ |
|-------|-------|------------|-------------|------------|------------|------------|
| **SearchAgent (Full)** | **0.0765** | 0.1104 | 0.1375 | **0.0821** | 0.2600 | **0.0200** |
| _Baselines_ | | | | | | |
| Random | 0.5331 | 0.5739 | 0.1983 | 0.8134 | 0.5100 | 0.0800 |
| Rule-Based | 0.2729 | 0.1840 | 0.2190 | 0.5522 | 0.2100 | 0.2300 |
| Naive LLM | 0.2526 | 0.1552 | 0.1722 | 0.5299 | 0.0400 | 0.1800 |
| _Prompting / agent methods_ | | | | | | |
| Skills | 0.2177 | 0.4065 | 0.2874 | 0.1194 | 0.0600 | 0.3300 |
| Dual-Prompt LLM | 0.2213 | 0.1552 | 0.2874 | 0.4403 | 0.2100 | 0.3300 |
| UXSim Hybrid | 0.2539 | 0.2004 | 0.2190 | 0.4776 | 0.2100 | 0.2300 |
| Few-Shot | 0.3258 | 0.1552 | 0.1991 | 0.7388 | 0.3100 | 0.0200 |
| Value-Guided | 0.3258 | 0.1552 | 0.0545 | 0.7388 | 0.5100 | 0.4200 |
| Controllable Persona | 0.3650 | 0.1552 | 0.2023 | 0.8507 | 0.4100 | 0.0700 |
| ICL Profile | 0.5269 | 0.1552 | 0.1555 | 1.3134 | 0.4600 | 0.0800 |
| Self-Reflection | 0.5530 | 0.1552 | 0.2388 | 1.3881 | 0.5100 | 0.2800 |
| ReAct | 0.5791 | 0.1552 | 0.2874 | 1.4627 | 0.5100 | 0.3300 |
| CoT | 0.5922 | 0.1552 | 0.2874 | 1.5000 | 0.5100 | 0.3300 |

SearchAgent (Full) achieves the **lowest CFS (0.0765)** of all 17 agents, outperforming every baseline and every alternative prompting/agent strategy.

### Ablation Study (ΔCFS vs. Full Model)

Removing any component degrades behavioral fidelity (higher CFS), confirming each contributes positively.

| Variant | CFS | ΔCFS | Key Observation |
|---------|-----|------|------------------|
| **SearchAgent (Full)** | **0.0765** | — | All components active |
| w/o Memory | 0.1156 | +0.0390 | Loses session-level consistency |
| w/o Persona | 0.1470 | +0.0705 | Trait-driven behavioral diversity removed |
| w/o Ad Model | 0.1500 | +0.0734 | Ad-interaction realism degrades |
| w/o Intent | 0.1677 | +0.0912 | Largest drop; intent evolution drives query/click dynamics |

### Per-Trait Behavioral Patterns (Full Model, 20 sessions)

| Trait | Value | Clicks | Ad Clicks | Abandon | Reform. |
|-------|-------|--------|-----------|---------|---------|
| **Patience** | low | 2.2 | 0.80 | 80% | 0% |
| | high | 1.1 | 0.25 | 75% | 50% |
| **Ad sensitivity** | ad_averse | 1.0 | 0.00 | 50% | 33% |
| | ad_tolerant | 1.8 | 0.89 | 89% | 33% |
| **Expertise** | novice | 1.8 | 0.38 | 88% | 13% |
| | expert | 1.0 | 0.14 | 57% | 43% |
| **Decision style** | impulsive | 1.9 | 0.71 | 71% | 29% |
| | deliberate | 1.2 | 0.00 | 80% | 20% |

## Configuration

Key parameters in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm.model_path` | `/root/models/Qwen/Qwen2-7B-Instruct` | Local LLM path |
| `llm.temperature` | 0.7 | Sampling temperature |
| `persona.ad_sensitivity` | 3 levels | ad_tolerant / ad_neutral / ad_averse |
| `persona.patience_level` | 3 levels | low / medium / high |
| `persona.decision_style` | 3 levels | impulsive / deliberate / price_sensitive |
| `persona.search_expertise` | 3 levels | novice / intermediate / expert |
| `simulation.num_sessions` | 100 | Sessions to simulate |
| `simulation.max_session_steps` | 20 | Max steps per session |
| `intent.initial_distribution` | [0.6, 0.3, 0.1] | exploratory / focused / transactional |

## Macro-Action Space

| Action | Description |
|--------|-------------|
| `click_organic` | Click on an organic search result |
| `click_ad` | Click on a sponsored advertisement |
| `skip` | Skip current results, continue scanning |
| `next_page` | Navigate to next SERP page |
| `reformulate` | Modify the current search query |
| `new_query` | Start a completely new query |
| `abandon` | Leave the search session (frustrated) |
| `end_session` | End session normally (satisfied) |

## Evaluation Metrics (5 core metrics)

All metrics are grounded in real search log studies. CFS is bounded in [0, 1] where lower = more realistic.

| Category | Metric | Reference | Method |
|----------|--------|-----------|--------|
| **Distributional** | Click Position JSD | TREC 2014: rank 1 = 61.8% (n=1,685 clicks) | Jensen-Shannon divergence vs. reference distribution |
| | Session Length JSD | TREC 2014: 38% single-query (n=1,257 sessions) | Jensen-Shannon divergence vs. reference distribution |
| **Rate-based** | Clicks/Session Gap | TREC 2014: 1.34; Jansen & Spink 2006: ~1.6 | Normalized absolute gap |
| | Abandon Rate Gap | TREC 2014: 49%; Stamou & Efthimiadis 2010: 35–50% | Normalized absolute gap |
| | Reformulation Rate Gap | Huang & Efthimiadis 2009: ~33% | Normalized absolute gap |
| **Composite** | **CFS** | Weighted sum of above 5 metrics | Bounded [0, 1], lower = better |

### Statistical Significance Testing

| Method | Use Case | Reference |
|--------|----------|-----------|
| Bootstrap CI (10k resamples) | Per-metric confidence intervals | Efron & Tibshirani (1993) |
| Paired permutation test | Pairwise agent comparison | Dror et al. (ACL 2018) |
| Cohen's d | Effect size quantification | Cohen (1988) |
| Wilcoxon signed-rank | Non-parametric paired test (n≥6) | Wilcoxon (1945) |

### Data Provenance

Reference distributions are extracted from **TREC Session Track 2014** (Carterette et al., 2014):
- 1,257 real search sessions with click-level annotations
- Reference statistics are defined in `searchagent/evaluation/reference_data.py`
- Cross-validated against 6 published large-scale log studies (see `reference_data.py` docstring)

## Differentiation from Existing Work

| | Agent4Rec | RecAgent | SUBER | **SearchAgent** |
|---|---|---|---|---|
| Scenario | Recommendation | Rec + Social | Recommendation | **Search + Ads** |
| LLM | GPT-3.5 API | GPT-3.5 API | GPTQ local | **Qwen2-7B local** |
| Ad modeling | No | No | No | **Yes** |
| Behavioral calibration | No | No | No | **5-metric CFS (real log grounded)** |
| Intent evolution | No | No | No | **Soft distribution** |
| Persona bias | Flat traits | Memory-based | Rating-based | **Trait-driven logprob bias** |
| Click selection | — | — | — | **LLM logprob × position bias hybrid** |
| Satisfaction | No | No | No | **LLM-rated + dwell time** |
| Baselines | Limited | None | None | **3 baselines + 5 ablations** |
| Analysis | Basic | Basic | Basic | **LaTeX tables + per-trait breakdowns** |
