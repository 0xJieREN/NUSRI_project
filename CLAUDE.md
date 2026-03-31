# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project Overview

Python 3.12 cryptocurrency research repository using QLib and LightGBM for BTCUSDT trading research.

The current direction is a config-driven workflow, and the mainline has already converged to:

- `config.toml` is the research configuration source of truth
- return-signal trading and probability-signal trading are separated
- QLib-first training and backtesting remain the execution backbone
- current mainline: `top23 + classification_72h_costaware(0.006) + rolling_2y_monthly + prob_conservative`

Latest comparison summary:
- `docs/research/2026-03-24-cost-aware-mainline-comparison.md`

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

# Train via config
uv run python -m scripts.training.lgbm_workflow --config config.toml --experiment-profile cost_aware_main

# Backtest via config
uv run python -m scripts.analysis.backtest_spot_strategy --pred-glob "/absolute/path/to/pred_*.pkl" --config config.toml --experiment-profile cost_aware_main

# Run probability-shell scan on the current mainline
uv run python -m scripts.analysis.run_72h_trade_tuning --predictions-root reports/costaware-prob-preds --config config.toml --experiment-profile cost_aware_main --year 2024
```

## Architecture

```
scripts/data/request_1h.py                → Download raw Binance data
scripts/data/clean_data.py                → Format CSV for QLib
scripts/data/dump_bin.py                  → Convert source CSV to QLib binary
scripts/training/lgbm_workflow.py         → Training entrypoint
nusri_project/training/lgbm_workflow.py   → Training workflow logic
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

**Current mainline settings:**
- factor set: `top23`
- label: `classification_72h_costaware`
- threshold: `positive_threshold = 0.006`
- model objective: `binary`
- training window: `rolling_2y_monthly`
- trade profile: `prob_conservative`

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
- Mainline comparison results are documented in `docs/research/2026-03-24-cost-aware-mainline-comparison.md`
- The current best-known mainline result files are:
  - `reports/cost-thr-006-2024/summary.json`
  - `reports/cost-thr-006-2025/summary.json`
- Historical comparison experiments such as `top15/top10`, `0.004/0.005`, and `18m/1y` are treated as completed analysis conclusions, not ongoing mainline configs
- Before writing custom backtest or portfolio-analysis code, check whether QLib already provides the needed capability through `qlib.backtest.backtest`, `qlib.contrib.evaluate.backtest_daily`, or `qlib.workflow.record_temp.PortAnaRecord`
- If QLib has a suitable built-in path, prefer configuring and integrating it over maintaining a parallel handwritten backtest stack
