# nusri-project

Python 3.12 cryptocurrency research repository built around QLib and LightGBM for BTCUSDT prediction, signal evaluation, and spot strategy backtesting.

The repository is now centered on a config-driven workflow with a fused regression signal and score-based execution:

- `config.toml` as the single source of truth for research profiles
- reusable fused signal training that emits continuous `pred_score`
- explicit separation between return, probability, and score execution semantics
- QLib-first training and backtesting entrypoints
- current best end-to-end experiment: `regression_fused_aggressive_v3_best`

Current best-stage summary:

- [docs/research/2026-04-01-regression-fused-best-stage-summary.md](/Users/jared/src/nusri-project/docs/research/2026-04-01-regression-fused-best-stage-summary.md)

## Structure

### Active entrypoints

- `scripts/data/` — data download and preparation commands
- `scripts/training/` — training commands
- `scripts/analysis/` — feature importance, backtest, and strategy scan commands

### Reusable logic

- `config.toml` — single source of truth for experiment profiles
- `nusri_project/config/` — factor config, runtime config schemas, and loaders
- `nusri_project/training/` — training workflow logic
- `nusri_project/strategy/` — return strategy, probability strategy, backtest wrapper, and scan helpers

### Data and outputs

- `data/raw/` — active raw market downloads
- `qlib_source_data/` — cleaned CSVs for QLib ingestion
- `qlib_data/my_crypto_data/` — QLib binary data
- `mlruns/` — MLflow experiment outputs

### Archived legacy files

- `archive/artifacts/` — archived generated CSV artifacts from earlier iterations
- `archive/legacy_test/` — archived old `test/` directory and Alpha158 notes/scripts

## Common Commands

### Download Binance 1h data

```bash
uv run python -m scripts.data.request_1h
```

### Clean raw CSV into QLib source format

```bash
uv run python -m scripts.data.clean_data --input data/raw/BTCUSDT_1h_binance_data.csv --output qlib_source_data/BTCUSDT.csv
```

### Dump QLib binary data

```bash
uv run python -m scripts.data.dump_bin dump_all --data_path qlib_source_data --qlib_dir qlib_data/my_crypto_data --freq 60min
```

### Train current fused signal from config

```bash
uv run python -m scripts.training.fused_signal_workflow --config config.toml --experiment-profile regression_fused_main --prediction-output-dir reports/fused-signal-preds/regression_fused_main
```

### Train single-component LightGBM workflow with legacy explicit flags

```bash
uv run python -m scripts.training.lgbm_workflow
```

Optional:

```bash
uv run python -m scripts.training.lgbm_workflow --feature-set top23 --run-mode rolling
```

### Export feature importance

```bash
uv run python -m scripts.analysis.dump_lgbm_feature_importance --importance-type gain --out reports/feature_importance/lgbm_feature_importance.csv --top 20
```

### Run current best spot backtest on saved fused prediction files

```bash
uv run python -m scripts.analysis.backtest_spot_strategy --pred-glob "reports/fused-signal-preds/regression_fused_main/pred_fused_2025*.pkl" --config config.toml --experiment-profile regression_fused_aggressive_v3_best --start-time "2025-01-01 00:00:00" --end-time "2025-12-31 23:00:00"
```

### Run 2024 validation backtest for the current best profile

```bash
uv run python -m scripts.analysis.backtest_spot_strategy --pred-glob "reports/fused-signal-preds/regression_fused_main/pred_fused_2024*.pkl" --config config.toml --experiment-profile regression_fused_aggressive_v3_best --start-time "2024-01-01 00:00:00" --end-time "2024-12-31 23:00:00"
```

### Run single-component / historical research utilities

```bash
uv run python -m scripts.analysis.run_cost_aware_label_round1 --predictions-root reports/costaware-preds --config config.toml --experiment-profile cost_aware_main --year 2025 --update-html
uv run python -m scripts.analysis.run_72h_trade_tuning --predictions-root reports/costaware-preds --config config.toml --experiment-profile regression_72h_main --year 2024 --update-html
uv run python -m scripts.analysis.run_phase2_baseline --mlruns-root ./mlruns --config config.toml --experiment-profile regression_72h_main --year 2024 --scan --update-html
```

## Notes

- QLib binary data must exist before training or backtesting.
- Spot backtests now use a QLib-first implementation with a custom single-asset order generator for fractional crypto sizing.
- Classification trading now uses `pred_prob` directly instead of mapping probability into pseudo-return.
- Fused signal experiments emit `pred_score`, and score-based strategies consume the same continuous interface.
- Return-signal, probability-signal, and score-signal strategies are now explicitly separated.
- Generated reports and temporary research outputs should not be committed unless explicitly needed.
