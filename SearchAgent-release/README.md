# SearchAgent

Behavior-constrained LLM user simulation for web search.

This repository implements the core SearchAgent pipeline described in the accompanying paper. Exact benchmark reproduction additionally requires the external benchmark corpora, frozen retrieval resources, split manifests, cached LLM outputs, and trained checkpoint used by the experiments. The simulator treats web search as a sequential decision process: at every step it constructs a structured user state, obtains a semantic macro-action prior from a frozen LLM, calibrates that prior with learned behavioral constraints, executes the selected action on a SERP, and feeds post-click utility and dwell time back into the next state.

## Core Pipeline

```text
Persona + Session Memory
        │
        ▼
Intent Distribution + Cognitive State
        │
        ▼
Structured Decision Context Ω_t
        │
        ├──────────────► LLM Macro-Action Prior
        │                       │
        │                       ▼
        │              Reliability Gate
        │                       +
        │              Prior-Orthogonal Residual
        │                       │
        ▼                       ▼
Behavior-Calibrated Macro-Action Posterior
        │
        ├── click_result ─────► Examination × Attractiveness Selector
        │                              │
        │                              ▼
        │                       Post-Click Utility
        │                              │
        │                              ▼
        │                         Dwell Time
        │
        ├── reformulate/new_query ─► Query Generation
        │
        └── skip/next_page/stop ───► Environment Transition
                                       │
                                       ▼
                              Memory / Intent / Cognitive Update
```

## Macro-Action Space

SearchAgent uses the seven macro-actions from the paper:

| Label | Action | Meaning |
|---|---|---|
| A | `click_result` | Click a visible result; the result-level selector chooses the concrete item |
| B | `skip` | Continue examining the current page without clicking the currently examined result |
| C | `next_page` | Request the next SERP page |
| D | `reformulate` | Preserve the current information need while revising the query |
| E | `new_query` | Explore another aspect of the same broader goal |
| F | `stop_satisfied` | Terminate because the need is sufficiently met |
| G | `stop_unsatisfied` | Terminate without sufficient satisfaction |

## Paper Baselines

The main comparison uses the twelve baseline families specified in Appendix C.3 of the paper. `random`, `rule_based`, and `naive_llm` remain available only as local diagnostics and are not paper baselines.

| CLI name | Paper method | Native output mapped by the harness |
|---|---|---|
| `react` | ReAct | Final `CLICK(i)`, `SKIP`, `NEXT`, query-change, or stop command; intermediate thoughts are ignored |
| `polyskill` | PolySkill | Selected skill plus arguments; navigation maps to skip/next-page, transition mode separates refinement from new search, finish status separates the two stop actions |
| `reflexion` | Reflexion | Only the post-reflection final command is executed, using the ReAct final-command mapping |
| `cot` | Chain-of-Thought | Final tagged label A-G is mapped through the fixed action map; rationale text has no environment effect |
| `tot` | Tree of Thoughts | Up to six internal branches may be considered; only the selected leaf's final tagged command is executed |
| `usersimcrs_v2` | UserSimCRS v2 | Inspect/accept with an item -> click, reject/continue -> skip, request-more -> next page, clarify -> reformulation, end -> satisfaction-conditioned stop |
| `cshi` | CSHI | Click/continue/refine/switch -> click/skip/reformulate/new query; finish uses the required `0.5` satisfaction threshold |
| `usimagent` | USimAgent | Click/browse/query/stop events map to unified actions; query changes use normalized Levenshtein distance `< 0.5` for reformulation and otherwise new query |
| `bases` | BASES | A predicted whole-session event sequence is consumed sequentially; query changes use the USimAgent edit-distance rule and terminal utility `>= 3` maps to satisfied stop |
| `agent4rec` | Agent4Rec | Item selection -> click; browsing -> skip or next page at a page boundary; preference refinement -> reformulation; goal shift -> new query; exit rating selects the stop state |
| `simuser` | SimUSER | Item choice/continue/request-more/refine/switch-goal/finish map directly to the corresponding unified actions |
| `convsearch_r1` | ConvSearch-R1 | Browse/click/finish map directly; a generated query retaining at least half of the canonical current-query tokens is a reformulation, otherwise a new query |

Every paper baseline is restricted to one final machine-readable environment command per step. An invalid command is retried once with the same interaction context; a second invalid response becomes `stop_unsatisfied`. Intermediate reasoning, reflection text, and tree branches are internal computation. The baseline wrapper enforces a maximum of six logical LLM interface invocations per environment step.

Appendix C.4 also reports equal-supervision controls such as PBM + Markov session, DBN + session controller, behavior cloning, and calibrated USimAgent/BASES variants. Their exact comparator architectures are not fully specified by the PDF beyond supervision, parameter budgets, validation targets, and trial counts, so this repository does not invent substitute implementations and does not claim to reproduce Table 18 without the released comparator artifacts.

## Structured State

The learned action calibrator consumes the 17-dimensional state used by the paper:

| Block | Dimensions | Implementation |
|---|---:|---|
| Persona | 3 | search expertise, decision style, patience |
| Cognitive state | 4 | need clarity, knowledge, frustration, exploration/focus |
| Intent distribution | 3 | exploratory, focused, transactional |
| Session memory statistics | 5 | normalized clicks, skips, last utility, page, step |
| Visible-SERP quality | 1 | expected score from Q0-Q10 |
| LLM uncertainty | 1 | normalized entropy of the macro-action prior |
| Total | 17 | `StateFeatures.dim()` |

The prompt memory retains the most recent 8 complete interaction records.

## Behavioral Calibration

The calibrated action posterior combines the frozen LLM semantic prior with two learned components:

1. A context-dependent reliability gate controls how much semantic evidence is retained.
2. A prior-orthogonal behavioral residual captures systematic behavioral deviations without collapsing into another confidence-scaling term.

The implemented learned components follow the paper configuration:

| Component | Architecture / setting |
|---|---|
| Behavioral calibrator | 17-64-32-7 MLP |
| Hidden normalization | LayerNorm |
| Hidden activation | GELU |
| Dropout | 0.1 after each hidden layer |
| Reliability gate | constrained generalized linear gate |
| `alpha_max` | 2.0 |
| Examination network | 5-16-1, GELU then softplus |
| Attractiveness network | 3-32-16-1, LayerNorm + GELU |
| Probability epsilon | `1e-8` |

The action residual is projected to be orthogonal to the centered LLM log-prior under the LLM probability measure before it is added to the gated semantic evidence.

## SERP Interaction

When the sampled macro-action is `click_result`, SearchAgent uses a hierarchical result selector.

The examination component receives only position-related and session-context features. The attractiveness component receives semantic preference, decision style, and text availability. Original SERP rank is removed from the LLM attractiveness prompt, candidates are deterministically permuted, and the mapped-back semantic probabilities are combined with monotone position-sensitive examination.

If every candidate is text-blind, the semantic candidate prior is exactly uniform before calibration. Opaque document or URL identifiers are kept opaque and are not assigned semantic meaning by the prompt.

## Intent and Cognitive-State Updates

Intent momentum is `0.8`. Transition targets use the coordinate order `(exploratory, focused, transactional)`:

| Action | Target |
|---|---|
| `click_result` | `(0.20, 0.60, 0.20)` |
| `skip` | `(0.45, 0.45, 0.10)` |
| `next_page` | `(0.55, 0.35, 0.10)` |
| `reformulate` | `(0.25, 0.65, 0.10)` |
| `new_query` | `(0.65, 0.25, 0.10)` |
| `stop_satisfied` | `(0.10, 0.75, 0.15)` |
| `stop_unsatisfied` | `(0.40, 0.45, 0.15)` |

Cognitive-state increments are fixed as follows:

| Transition | Increment `(clarity, knowledge, frustration, focus)` |
|---|---|
| Click with normalized utility `u` | `(0.08u, 0.12u, 0.05(1-u)-0.08u, 0.03)` |
| `skip` | `(-0.01, 0, 0.03, -0.02)` |
| `next_page` | `(-0.02, 0, 0.05, -0.04)` |
| `reformulate` | `(0.06, 0, -0.02, 0.04)` |
| `new_query` | `(-0.03, 0, -0.01, -0.06)` |

All cognitive coordinates are clipped to `[0, 1]` after each update.

## Post-Click Feedback

Post-click utility is elicited on `U1-U5` and converted to an expected utility in `[1, 5]`.

Dwell time follows the paper configuration:

- Base dwell times for utility levels 1-5: `[5, 15, 35, 70, 120]` seconds
- Log-normal multiplicative noise: `sigma = 0.35`

The utility and sampled dwell time are written back into the session record and influence subsequent state updates.

## LLM Interfaces

The default requested model is `deepseek-v4-pro`. The implementation uses the DeepSeek OpenAI-compatible chat endpoint in JSON Output mode and applies exact local schema validation before accepting a response.

| Interface | Output | Visible inputs | Excluded information |
|---|---|---|---|
| Macro-action prior | probabilities over A-G | persona, current query, visible SERP, recent history, intent, cognitive state, session metadata | calibrated posterior, future actions, evaluation statistics |
| SERP quality | probabilities over Q0-Q10 | current query, visible SERP | persona, history, intent, hidden relevance, user action |
| Candidate attractiveness | probabilities over candidate labels | persona, current query, rank-free candidate content | original rank, examination probability, hidden relevance |
| Post-click utility | probabilities over U1-U5 | persona, current query, clicked-result visible fields | future outcomes, calibrated probabilities, hidden relevance |
| Query generation | one JSON query object | persona, current query, recent history, intent, transition mode | future SERPs, evaluation labels, hidden relevance |

Finite-label requests use temperature zero and a 256-token output cap. Label order is deterministically rotated before each request and mapped back after scoring.

Probability outputs are validated, smoothed by `epsilon`, and renormalized. A malformed probability response is retried once with the same semantic prompt plus a format-only reminder; a second failure returns a uniform simplex. Query generation is also retried once; a second failure keeps the current query. Generated queries must be non-empty, single-line, and at most 32 whitespace-delimited tokens.

## Evaluation

The primary aggregate metric is Population Fidelity Discrepancy (PFD). It averages five bootstrap-normalized discrepancies:

1. clicked-rank Jensen-Shannon divergence
2. session-length Jensen-Shannon divergence
3. clicks-per-session absolute gap
4. abandonment-rate absolute gap
5. reformulation-rate absolute gap

Lower PFD is better. `CFS` is retained in the code only as a compatibility alias for `PFD`.

Matched-SERP replay additionally reports micro-averaged clicked-rank precision, recall, F1, and abandonment agreement.

## Training Configuration

The learned action and click components are trained with the paper settings:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Betas | `(0.9, 0.999)` |
| Weight decay | `1e-4` |
| Action batch size | 512 |
| Click batch size | 256 |
| Gradient clipping | 1.0 |
| Maximum epochs | 100 |
| Early-stop patience | 10 validation epochs |
| Minimum validation improvement | `1e-4` |
| Validation objective | equally weighted action NLL and click NLL |

The best validation checkpoint is saved rather than the final epoch.

## Supported Data Paths

The repository includes loaders and replay support for the benchmark families used by the paper:

- TREC Session Track 2012
- TREC Session Track 2013
- TREC Session Track 2014
- Yandex session logs
- ORCAS query-document click data

Benchmark files are not bundled with this source archive. Default paths expected by the calibration script are defined in `searchagent/training/train_calibrator.py` under `data/search_logs/`. The archive also does not bundle the frozen ClueWeb09/ClueWeb12 retrieval indexes, MS MARCO document resources used for ORCAS field completion, released split manifests, cached provider outputs, or a trained calibrator checkpoint. Synthetic execution and local replay can therefore validate the software path, but they are not substitutes for reproducing the paper's reported benchmark numbers.

The experiment runner also supports synthetic SERPs for smoke testing and local development.

## Project Structure

```text
SearchAgent-release/
├── configs/
│   ├── default.yaml
│   ├── trec2012.yaml
│   ├── trec2013.yaml
│   ├── trec2014.yaml
│   ├── yandex.yaml
│   └── orcas.yaml
├── scripts/
│   ├── run_simulation.py
│   ├── run_experiment.py
│   └── evaluate.py
├── searchagent/
│   ├── agent/
│   │   ├── baselines.py
│   │   ├── calibration.py
│   │   ├── llm_backbone.py
│   │   ├── memory.py
│   │   ├── persona.py
│   │   ├── prompts.py
│   │   └── search_agent.py
│   ├── environment/
│   │   ├── search_env.py
│   │   ├── serp.py
│   │   └── serp_templates.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── reference_data.py
│   │   └── statistical_tests.py
│   ├── simulation/
│   │   └── session_sim.py
│   └── training/
│       ├── data_builder.py
│       ├── losses.py
│       └── train_calibrator.py
└── tests/
    ├── test_smoke.py
    └── test_extended.py
```

## Installation

Python 3.9 or newer is required.

```bash
pip install -e .
pip install requests pytest
```

For LLM-backed execution, configure the API key and endpoint:

```bash
export DEEPSEEK_API_KEY=sk-your-key-here
export DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
```

`DEEPSEEK_API_URL` is optional; the value above is the default. The requested model name is configured in `configs/*.yaml` under `llm.model_name`.

## Smoke Test

The test suite does not require a real API key and does not issue live LLM requests.

```bash
python -m pytest -q tests/test_smoke.py
```

To run the complete local test suite:

```bash
python -m pytest -q
```

## Build the LLM Prior Cache

Calibration training consumes cached LLM probabilities for logged states. After placing the benchmark data at the expected paths and setting `DEEPSEEK_API_KEY`, build the cache first:

```bash
python -m searchagent.training.train_calibrator build-cache \
  --train trec2012 trec2013 \
  --val trec2014 \
  --out output/calib
```

## Train the Calibrator

```bash
python -m searchagent.training.train_calibrator train \
  --train trec2012 trec2013 \
  --val trec2014 \
  --config configs/default.yaml \
  --out output/calib \
  --seed 42
```

The resulting checkpoint is written to:

```text
output/calib/calibrator.pt
```

## Run a Synthetic Session Simulation

`configs/default.yaml` points to `output/calib/calibrator.pt`. If that checkpoint exists, it is loaded automatically. If it does not exist, simulation falls back to the structured LLM prior without the learned calibrator. An LLM API key is still required for `full`, `full_pp`, and other LLM-backed agents.

```bash
python scripts/run_simulation.py \
  --config configs/default.yaml \
  --num_sessions 10 \
  --output output/simulation_results.json
```

## Run Experiments

Synthetic mode:

```bash
python scripts/run_experiment.py \
  --config configs/default.yaml \
  --agents full_pp react polyskill reflexion cot tot usersimcrs_v2 cshi usimagent bases agent4rec simuser convsearch_r1 \
  --num_sessions 20 \
  --seeds 42 123 456 \
  --output_dir output/experiments
```

Matched-SERP replay on TREC:

```bash
python scripts/run_experiment.py \
  --config configs/trec2014.yaml \
  --agents full_pp \
  --mode replay \
  --trec_data data/search_logs/trec_session2014.xml \
  --calibrator output/calib/calibrator.pt \
  --seeds 42 123 456 \
  --output_dir output/trec2014_replay
```

SERP ranking sensitivity can be run with `--serp_perturbation shuffle` or `--serp_perturbation reverse`.

Learned-component ablations can be selected with:

```text
--ablate no_f
--ablate no_gate
--ablate no_click
```

The runner accepts both `full` and `full_pp`; both construct the same SearchAgent class, and `full` is retained as a compatibility alias.

## Evaluate Saved Sessions

```bash
python scripts/evaluate.py output/simulation_results.json \
  --output output/simulation_results.metrics.json
```

## Reproducibility Controls

The implementation includes the following reproducibility and isolation controls from the paper:

- deterministic label rotation for finite-label LLM requests
- deterministic rank-independent candidate permutation
- strict field-level availability masks and `text_blind` handling
- opaque identifier isolation
- JSON output with exact local schema validation
- request retry and fail-closed fallbacks
- run-manifest recording through `SEARCHAGENT_RUN_MANIFEST`
- fixed memory length of 8
- maximum session budget of 20 steps
- maximum 3 SERP pages with `max_results_per_page=10` in the benchmark configuration
- ORCAS one-step configuration in `configs/orcas.yaml`

For request pacing, `SEARCHAGENT_MIN_REQUEST_GAP` can be set to a minimum number of seconds between API calls.
