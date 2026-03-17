# Config-Driven Research Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the repository into a config-driven research workflow with explicit separation between regression-signal trading and probability-signal trading, starting with a safe phase-1 foundation.

**Architecture:** Introduce a single `config.toml` source of truth plus runtime config loaders, then progressively route training and backtesting through typed runtime configs. Preserve existing QLib-first backtesting, but split regression and classification trading semantics so classification consumes probabilities directly instead of mapped pseudo-returns.

**Tech Stack:** Python 3.12, `uv`, QLib, LightGBM, TOML via stdlib `tomllib`, unittest

---

### Task 1: Add Config Schema and Loader Foundation

**Files:**
- Create: `config.toml`
- Create: `nusri_project/config/schemas.py`
- Create: `nusri_project/config/runtime_config.py`
- Modify: `tests/test_label_optimization_round1.py`
- Create: `tests/test_runtime_config.py`

- [ ] **Step 1: Write the failing tests for config parsing**

Add tests covering:
- default profile resolution
- experiment profile composition
- classification label config parsing
- probability trade config parsing
- invalid config validation

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_runtime_config -v
```

Expected:
- FAIL due to missing module or missing config loader functions

- [ ] **Step 3: Write minimal config files and schema dataclasses**

Implement:
- `config.toml` with first-pass data/factor/label/model/training/trading/experiment profiles
- dataclasses for each config layer
- `load_runtime_config(...)` and helpers

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_runtime_config tests.test_label_optimization_round1 -v
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add config.toml nusri_project/config/schemas.py nusri_project/config/runtime_config.py tests/test_runtime_config.py tests/test_label_optimization_round1.py
git commit -m "feat: add config-driven runtime config foundation"
```

### Task 2: Route Training Workflow Through Runtime Config

**Files:**
- Modify: `nusri_project/training/lgbm_workflow.py`
- Create: `nusri_project/training/label_factory.py`
- Create: `nusri_project/training/model_factory.py`
- Modify: `scripts/training/lgbm_workflow.py`
- Create: `tests/test_lgbm_workflow_config.py`

- [ ] **Step 1: Write the failing tests for config-driven training selection**

Add tests covering:
- loading regression label expr from config
- loading cost-aware classification label config from config
- selecting LightGBM objective from config
- artifact naming includes label mode semantics correctly

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_lgbm_workflow_config -v
```

Expected:
- FAIL due to missing factories or config-driven entrypoints

- [ ] **Step 3: Implement config-driven training factories**

Implement minimal changes so:
- label definitions come from runtime config
- model objective/hyperparams come from runtime config
- workflow can resolve experiment profile into runnable training config

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_lgbm_workflow_config tests.test_label_optimization_round1 -v
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add nusri_project/training/lgbm_workflow.py nusri_project/training/label_factory.py nusri_project/training/model_factory.py scripts/training/lgbm_workflow.py tests/test_lgbm_workflow_config.py
git commit -m "feat: route training workflow through runtime config"
```

### Task 3: Split Regression and Probability Trading Semantics

**Files:**
- Create: `nusri_project/strategy/return_signal_strategy.py`
- Create: `nusri_project/strategy/probability_signal_strategy.py`
- Modify: `nusri_project/strategy/backtest_spot_strategy.py`
- Modify: `nusri_project/strategy/phase2_strategy_research.py`
- Modify: `tests/test_backtest_spot_strategy.py`
- Create: `tests/test_probability_signal_strategy.py`

- [ ] **Step 1: Write the failing tests for probability trading semantics**

Add tests covering:
- classification predictions use `pred_prob`
- probability thresholds decide entry/full/exit
- `transform_prediction_score` is no longer used
- de-risk validation warns or rejects `de_risk_position == max_position`

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_probability_signal_strategy tests.test_backtest_spot_strategy -v
```

Expected:
- FAIL because probability strategy path does not exist yet

- [ ] **Step 3: Implement explicit probability strategy path**

Implement:
- separate probability strategy class or function path
- classification signal field `pred_prob`
- backtest runner chooses strategy by signal kind
- remove `transform_prediction_score` from active execution path

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_probability_signal_strategy tests.test_backtest_spot_strategy tests.test_phase2_strategy_research -v
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add nusri_project/strategy/return_signal_strategy.py nusri_project/strategy/probability_signal_strategy.py nusri_project/strategy/backtest_spot_strategy.py nusri_project/strategy/phase2_strategy_research.py tests/test_probability_signal_strategy.py tests/test_backtest_spot_strategy.py
git commit -m "feat: split return and probability trading strategies"
```

### Task 4: Make Analysis Entrypoints Config-Driven

**Files:**
- Modify: `scripts/analysis/run_label_optimization_round1.py`
- Modify: `scripts/analysis/run_cost_aware_label_round1.py`
- Modify: `scripts/analysis/run_72h_trade_tuning.py`
- Modify: `scripts/analysis/run_phase2_baseline.py`
- Modify: `scripts/analysis/backtest_spot_strategy.py`
- Create: `tests/test_analysis_entrypoints_config.py`

- [ ] **Step 1: Write the failing tests for config-driven CLI entrypoints**

Add tests covering:
- each entrypoint accepts `--config`
- each entrypoint accepts experiment/profile selection
- legacy direct flags still work only where intentionally preserved

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_analysis_entrypoints_config -v
```

Expected:
- FAIL due to missing arguments or config resolution

- [ ] **Step 3: Implement config-driven entrypoint wiring**

Implement minimal wiring so analysis entrypoints can load runtime config and run named experiments without local hard-coded profile constants.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_analysis_entrypoints_config tests.test_html_reports -v
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/run_label_optimization_round1.py scripts/analysis/run_cost_aware_label_round1.py scripts/analysis/run_72h_trade_tuning.py scripts/analysis/run_phase2_baseline.py scripts/analysis/backtest_spot_strategy.py tests/test_analysis_entrypoints_config.py
git commit -m "feat: make analysis entrypoints config-driven"
```

### Task 5: Clean Up Legacy Defaults and Document the New Workflow

**Files:**
- Modify: `nusri_project/strategy/strategy_config.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `docs/plans/2026-03-17-config-driven-research-design.md`
- Create: `docs/research/2026-03-17-config-driven-usage.md`

- [ ] **Step 1: Write the failing tests or assertions for legacy-default cleanup**

Add tests if practical, otherwise add explicit validation checks to ensure:
- no active classification path depends on pseudo-return mapping
- legacy default profiles do not silently override `config.toml`

- [ ] **Step 2: Run targeted verification and confirm current failures**

Run targeted commands needed to demonstrate remaining legacy behavior before cleanup.

- [ ] **Step 3: Implement cleanup and docs**

Update docs to explain:
- config layout
- experiment profile selection
- regression vs probability trading shells
- recommended current mainline profile

- [ ] **Step 4: Run final verification**

Run:

```bash
uv run python -m unittest tests.test_runtime_config tests.test_lgbm_workflow_config tests.test_probability_signal_strategy tests.test_analysis_entrypoints_config tests.test_backtest_spot_strategy tests.test_phase2_strategy_research tests.test_label_optimization_round1 tests.test_html_reports -v
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add nusri_project/strategy/strategy_config.py README.md AGENTS.md CLAUDE.md docs/plans/2026-03-17-config-driven-research-design.md docs/research/2026-03-17-config-driven-usage.md
git commit -m "docs: finalize config-driven workflow"
```
