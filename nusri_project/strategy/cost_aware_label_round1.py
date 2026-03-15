from __future__ import annotations

from pathlib import Path

import pandas as pd

from nusri_project.strategy.label_optimization_round1 import build_round1_trading_shells
from nusri_project.strategy.phase2_strategy_research import run_strategy_config
from nusri_project.strategy.strategy_config import SpotStrategyConfig


def build_cost_aware_round1_modes() -> list[str]:
    return ["regression_72h", "classification_72h_costaware"]


def build_cost_aware_round1_matrix() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label_mode in build_cost_aware_round1_modes():
        for shell_name in build_round1_trading_shells():
            rows.append({"label_mode": label_mode, "shell_name": shell_name})
    return rows


def find_cost_aware_prediction_files(
    prediction_dir: Path,
    *,
    label_mode: str,
    label_horizon_hours: int,
    year: int,
) -> list[Path]:
    mode_pattern = f"pred_{label_mode}_{label_horizon_hours}h_{year}[0-1][0-9].pkl"
    files = sorted(prediction_dir.glob(mode_pattern), key=lambda path: path.name)
    if files:
        return files

    if label_mode == "regression_72h":
        legacy_pattern = f"pred_{label_horizon_hours}h_{year}[0-1][0-9].pkl"
        return sorted(prediction_dir.glob(legacy_pattern), key=lambda path: path.name)
    return []


def evaluate_cost_aware_round1(
    *,
    predictions_root: Path,
    output_root: Path,
    provider_uri: str,
    year: int,
    label_horizon_hours: int = 72,
    instrument: str = "BTCUSDT",
    freq: str = "60min",
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.001,
    min_cost: float = 0.0,
    deal_price: str = "close",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    shells = build_round1_trading_shells()

    for item in build_cost_aware_round1_matrix():
        label_mode = item["label_mode"]
        shell_name = item["shell_name"]
        prediction_files = find_cost_aware_prediction_files(
            predictions_root / label_mode,
            label_mode=label_mode,
            label_horizon_hours=label_horizon_hours,
            year=year,
        )
        if not prediction_files:
            continue

        shell = shells[shell_name]
        config = SpotStrategyConfig(
            provider_uri=provider_uri,
            instrument=instrument,
            start_time=f"{year}-01-01 00:00:00",
            end_time=f"{year}-12-31 23:00:00",
            freq=freq,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            min_cost=min_cost,
            deal_price=deal_price,
            entry_threshold=shell["entry_threshold"],
            exit_threshold=shell["exit_threshold"],
            full_position_threshold=shell["full_position_threshold"],
            max_position=shell["max_position"],
            min_holding_hours=shell["min_holding_hours"],
            cooldown_hours=shell["cooldown_hours"],
            drawdown_de_risk_threshold=shell["drawdown_de_risk_threshold"],
            de_risk_position=shell["de_risk_position"],
        )
        summary = run_strategy_config(
            prediction_files,
            config,
            output_dir=output_root / label_mode / shell_name,
        )
        rows.append(
            {
                "label_mode": label_mode,
                "shell_name": shell_name,
                **shell,
                **summary,
            }
        )
    return pd.DataFrame(rows)
