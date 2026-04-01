# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project Overview

Python 3.12 cryptocurrency research repository using QLib and LightGBM for BTCUSDT trading research.

The current direction is a config-driven workflow, and the mainline has already converged to:

- `config.toml` is the research configuration source of truth
- fused signal training and score-based execution are first-class paths
- return-signal, probability-signal, and score-signal trading are separated
- QLib-first training and backtesting remain the execution backbone
- current best stage: `top23 + regression_fused_main + rolling_24m_halflife_6m + score_regression_aggressive_v3_best`

Latest current-stage summary:
- `docs/research/2026-04-01-regression-fused-best-stage-summary.md`

## Environment

- Python: 3.12 (see `.python-version`)
- Package manager: **uv** (required)
- Install dependencies: `uv sync`
- Run scripts: `uv run python <script>.py [args...]`

## Common Commands

```bash
# Download Binance 1h data
uv run python -m scripts.data.request_1h

# Clean CSV → QLib source CSV
uv run python -m scripts.data.clean_data --input data/raw/BTCUSDT_1h_binance_data.csv --output qlib_source_data/BTCUSDT.csv

# QLib source CSV → QLib binary
uv run python -m scripts.data.dump_bin dump_all --data_path qlib_source_data --qlib_dir qlib_data/my_crypto_data --freq 60min

# Train current fused signal
uv run python -m scripts.training.fused_signal_workflow --config config.toml --experiment-profile regression_fused_main --prediction-output-dir reports/fused-signal-preds/regression_fused_main

# Backtest current best 2025 profile
uv run python -m scripts.analysis.backtest_spot_strategy --pred-glob "reports/fused-signal-preds/regression_fused_main/pred_fused_2025*.pkl" --config config.toml --experiment-profile regression_fused_aggressive_v3_best --start-time "2025-01-01 00:00:00" --end-time "2025-12-31 23:00:00"

# Historical probability-shell scan utility
uv run python -m scripts.analysis.run_72h_trade_tuning --predictions-root reports/costaware-prob-preds --config config.toml --experiment-profile cost_aware_main --year 2024
```

## Architecture

```
scripts/data/request_1h.py                → Download raw Binance data
scripts/data/clean_data.py                → Format CSV for QLib
scripts/data/dump_bin.py                  → Convert source CSV to QLib binary
scripts/training/lgbm_workflow.py         → Single-component training entrypoint
scripts/training/fused_signal_workflow.py → Fused signal training entrypoint
nusri_project/training/lgbm_workflow.py   → Training workflow logic
nusri_project/training/fused_signal_workflow.py → Fused orchestration logic
scripts/analysis/backtest_spot_strategy.py → Spot backtest entrypoint
nusri_project/strategy/*                  → Strategy and scan helpers
```

**Key modules:**
- `nusri_project/config/alpha261_config.py` — Alpha261 and Top23 feature configurations
- `nusri_project/training/lgbm_workflow.py` — Model training with config-driven single/rolling modes
- `nusri_project/strategy/backtest_spot_strategy.py` — QLib-first spot backtest wrapper
- `nusri_project/strategy/phase2_strategy_research.py` — phase 2 scan helpers
- `qlib_data/my_crypto_data/` — QLib binary data directory

**Primary config source:**
- `config.toml`

**Important strategy split:**
- Return mode emits `pred_return` and uses return thresholds
- Classification mode emits `pred_prob` and uses probability thresholds
- Fused / score mode emits `pred_score` and uses continuous score thresholds

**Current best-stage settings:**
- factor set: `top23`
- fused components: `reg_24h`, `reg_72h`
- fusion profile: `regression_fused_main`
- model objective: `mse`
- training window: `rolling_24m_halflife_6m`
- trade profile: `score_regression_aggressive_v3_best`
- experiment: `regression_fused_aggressive_v3_best`

## Data Paths

- Raw downloads: `data/raw/`
- QLib source CSV: `qlib_source_data/`
- QLib binary: `qlib_data/my_crypto_data/`
- MLflow logs: `mlruns/`
- Archived artifacts: `archive/artifacts/`

## Important Notes

- QLib binary data must exist before training (run the pipeline in order)
- High-frequency time format: `%Y-%m-%d %H:%M:%S`
- Alpha261 factor names must be unique (raises `ValueError` on duplicates)
- Do not commit: `qlib_data/`, `mlruns/`, large CSV files (see `.gitignore`)
- `nusri_project/strategy/strategy_config.py` is now a runtime transport/compatibility layer, not the source of truth for research defaults
- Current best-stage results are documented in `docs/research/2026-04-01-regression-fused-best-stage-summary.md`
- Historical comparison experiments such as `top15/top10`, `0.004/0.005`, `18m/1y`, and the old cost-aware probability mainline are treated as completed analysis conclusions, not ongoing mainline configs
- Before writing custom backtest or portfolio-analysis code, check whether QLib already provides the needed capability through `qlib.backtest.backtest`, `qlib.contrib.evaluate.backtest_daily`, or `qlib.workflow.record_temp.PortAnaRecord`
- If QLib has a suitable built-in path, prefer configuring and integrating it over maintaining a parallel handwritten backtest stack
