# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python 3.12 cryptocurrency price prediction project using QLib framework and LightGBM for BTCUSDT trading. Script-driven workflow (not a standard Python package).

## Environment

- Python: 3.12 (see `.python-version`)
- Package manager: **uv** (required)
- Install dependencies: `uv sync`
- Run scripts: `uv run python <script>.py [args...]`

## Common Commands

```bash
# Download Binance 1h data
uv run python request_1h.py

# Clean CSV → QLib source CSV
uv run python clean_data.py --input BTCUSDT_1h_binance_data.csv --output qlib_source_data/BTCUSDT.csv

# QLib source CSV → QLib binary
uv run python dump_bin.py dump_all --data_path qlib_source_data --qlib_dir qlib_data/my_crypto_data --freq 60min

# Train LightGBM model (single or rolling mode)
uv run python LGBM_workflow.py

# Export feature importance
uv run python dump_lgbm_feature_importance.py --importance-type gain --out lgbm_feature_importance.csv --top 20

# Run smoke test (verify QLib data loads)
uv run python test/test_qlib.py
```

## Architecture

```
request_1h.py        → Download raw Binance data (CSV)
    ↓
clean_data.py        → Format CSV for QLib (add symbol, date format)
    ↓
dump_bin.py          → Convert to QLib binary format
    ↓
LGBM_workflow.py    → Main training pipeline (QLib + LightGBM)
    ↓
mlruns/              → MLflow tracking output
```

**Key modules:**
- `alpha261_config.py` — Alpha261 and Top23 feature configurations
- `LGBM_workflow.py` — Model training with single/rolling modes, configurable feature sets
- `qlib_data/my_crypto_data/` — QLib binary data directory

**Configurable options in LGBM_workflow.py:**
- `FEATURE_SET`: "alpha261" or "top23"
- `RUN_MODE`: "single" or "rolling"

## Data Paths

- Raw data: `qlib_source_data/`
- QLib binary: `qlib_data/my_crypto_data/`
- MLflow logs: `mlruns/`
- Generated CSV: `BTCUSDT_1h_binance_data.csv`

## Important Notes

- QLib binary data must exist before training (run the pipeline in order)
- High-frequency time format: `%Y-%m-%d %H:%M:%S`
- Alpha261 factor names must be unique (raises `ValueError` on duplicates)
- Do not commit: `qlib_data/`, `mlruns/`, large CSV files (see `.gitignore`)
- Before writing custom backtest or portfolio-analysis code, check whether QLib already provides the needed capability through `qlib.backtest.backtest`, `qlib.contrib.evaluate.backtest_daily`, or `qlib.workflow.record_temp.PortAnaRecord`
- If QLib has a suitable built-in path, prefer configuring and integrating it over maintaining a parallel handwritten backtest stack
