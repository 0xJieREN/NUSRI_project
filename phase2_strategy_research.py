from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path

import pandas as pd

from backtest_spot_strategy import (
    compute_monthly_returns,
    load_prediction_frames,
    prepare_signal_frame,
    run_qlib_backtest,
    summarize_report,
)
from strategy_config import SpotStrategyConfig


def find_prediction_files(mlruns_root: Path, year: int) -> list[Path]:
    matches = sorted(
        mlruns_root.rglob(f"pred_{year}[0-1][0-9].pkl"),
        key=lambda path: path.name,
    )
    return [path for path in matches if path.stem[5:9] == str(year)]


def build_parameter_grid(
    *,
    entry_thresholds: list[float],
    exit_thresholds: list[float],
    full_position_thresholds: list[float],
    min_holding_hours_list: list[int],
    cooldown_hours_list: list[int],
    drawdown_thresholds: list[float],
    de_risk_positions: list[float],
) -> list[dict]:
    candidates: list[dict] = []
    for values in product(
        entry_thresholds,
        exit_thresholds,
        full_position_thresholds,
        min_holding_hours_list,
        cooldown_hours_list,
        drawdown_thresholds,
        de_risk_positions,
    ):
        (
            entry_threshold,
            exit_threshold,
            full_position_threshold,
            min_holding_hours,
            cooldown_hours,
            drawdown_de_risk_threshold,
            de_risk_position,
        ) = values

        if full_position_threshold < entry_threshold:
            continue
        if entry_threshold < exit_threshold:
            continue

        candidates.append(
            {
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "full_position_threshold": full_position_threshold,
                "min_holding_hours": min_holding_hours,
                "cooldown_hours": cooldown_hours,
                "drawdown_de_risk_threshold": drawdown_de_risk_threshold,
                "de_risk_position": de_risk_position,
            }
        )
    return candidates


def rank_scan_results(
    frame: pd.DataFrame,
    *,
    min_annualized_return: float = 0.04,
    max_drawdown: float = 0.10,
) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["meets_constraints"] = (
        (ranked["annualized_return"] >= min_annualized_return)
        & (ranked["max_drawdown"] <= max_drawdown)
    )
    ranked = ranked.sort_values(
        by=["meets_constraints", "sharpe", "calmar", "annualized_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return ranked


def write_backtest_outputs(
    *,
    output_dir: Path,
    report: pd.DataFrame,
    monthly: pd.DataFrame,
    summary: dict,
    positions,
    indicators: pd.DataFrame | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "report.csv")
    monthly.to_csv(output_dir / "monthly_returns.csv")
    pd.to_pickle(positions, output_dir / "positions.pkl")
    if indicators is not None:
        indicators.to_csv(output_dir / "indicators.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def run_strategy_config(
    prediction_files: list[Path],
    config: SpotStrategyConfig,
    *,
    output_dir: Path | None = None,
) -> dict:
    combined = load_prediction_frames(prediction_files)
    signal = prepare_signal_frame(combined, instrument=config.instrument)
    report, positions, indicators = run_qlib_backtest(signal, config)
    summary = summarize_report(report, annualization_periods=config.annualization_periods)
    monthly = compute_monthly_returns(report)

    if output_dir is not None:
        write_backtest_outputs(
            output_dir=output_dir,
            report=report,
            monthly=monthly,
            summary=summary,
            positions=positions,
            indicators=indicators,
        )
    return summary


def run_parameter_scan(
    prediction_files: list[Path],
    base_config: SpotStrategyConfig,
    parameter_grid: list[dict],
) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate_id, params in enumerate(parameter_grid, start=1):
        config = replace(base_config, **params)
        summary = run_strategy_config(prediction_files, config)
        rows.append({"candidate_id": candidate_id, **params, **summary})
    return pd.DataFrame(rows)


def baseline_summary_row(config: SpotStrategyConfig, summary: dict) -> dict:
    return {**asdict(config), **summary}
