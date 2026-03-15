from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path

import pandas as pd

from nusri_project.strategy.backtest_spot_strategy import (
    compute_monthly_returns,
    load_prediction_frames,
    prepare_signal_frame,
    run_qlib_backtest,
    summarize_report,
)
from nusri_project.strategy.strategy_config import SpotStrategyConfig


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
    max_positions: list[float] | None = None,
    min_holding_hours_list: list[int],
    cooldown_hours_list: list[int],
    drawdown_thresholds: list[float],
    de_risk_positions: list[float],
) -> list[dict]:
    if max_positions is None:
        max_positions = [1.0]

    candidates: list[dict] = []
    for values in product(
        entry_thresholds,
        exit_thresholds,
        full_position_thresholds,
        max_positions,
        min_holding_hours_list,
        cooldown_hours_list,
        drawdown_thresholds,
        de_risk_positions,
    ):
        (
            entry_threshold,
            exit_threshold,
            full_position_threshold,
            max_position,
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
                "max_position": max_position,
                "min_holding_hours": min_holding_hours,
                "cooldown_hours": cooldown_hours,
                "drawdown_de_risk_threshold": drawdown_de_risk_threshold,
                "de_risk_position": de_risk_position,
            }
        )
    return candidates


def build_scan_profile(profile: str) -> list[dict]:
    if profile == "small":
        return build_parameter_grid(
            entry_thresholds=[0.0001, 0.0005],
            exit_thresholds=[-0.0001, 0.0],
            full_position_thresholds=[0.0002, 0.001],
            max_positions=[1.0],
            min_holding_hours_list=[1, 24],
            cooldown_hours_list=[1, 12],
            drawdown_thresholds=[0.08],
            de_risk_positions=[0.5],
        )
    if profile == "conservative":
        return build_parameter_grid(
            entry_thresholds=[0.001, 0.002, 0.003],
            exit_thresholds=[0.0],
            full_position_thresholds=[0.002, 0.004],
            max_positions=[0.15, 0.25, 0.35, 0.5],
            min_holding_hours_list=[24, 48],
            cooldown_hours_list=[12],
            drawdown_thresholds=[0.02, 0.05],
            de_risk_positions=[0.0, 0.25],
        )
    if profile == "conservative_fast":
        return build_parameter_grid(
            entry_thresholds=[0.0015, 0.003],
            exit_thresholds=[0.0],
            full_position_thresholds=[0.003],
            max_positions=[0.15, 0.25],
            min_holding_hours_list=[24, 48],
            cooldown_hours_list=[12],
            drawdown_thresholds=[0.02, 0.05],
            de_risk_positions=[0.0, 0.25],
        )
    if profile == "label72_trade_tuning":
        threshold_pairs = [
            (0.0015, 0.003),
            (0.003, 0.005),
            (0.005, 0.008),
            (0.008, 0.012),
        ]
        risk_pairs = [
            (0.02, 0.0),
            (0.02, 0.10),
            (0.04, 0.20),
            (0.06, 0.25),
        ]
        candidates: list[dict] = []
        for entry_threshold, full_position_threshold in threshold_pairs:
            for max_position in [0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
                for min_holding_hours in [48, 72, 96]:
                    for drawdown_de_risk_threshold, de_risk_position in risk_pairs:
                        candidates.append(
                            {
                                "entry_threshold": entry_threshold,
                                "exit_threshold": 0.0,
                                "full_position_threshold": full_position_threshold,
                                "max_position": max_position,
                                "min_holding_hours": min_holding_hours,
                                "cooldown_hours": 12,
                                "drawdown_de_risk_threshold": drawdown_de_risk_threshold,
                                "de_risk_position": de_risk_position,
                            }
                        )
        return candidates
    if profile == "label72_trade_tuning_fast":
        threshold_pairs = [
            (0.0015, 0.003),
            (0.003, 0.005),
            (0.005, 0.008),
        ]
        risk_pairs = [
            (0.02, 0.0),
            (0.02, 0.25),
            (0.04, 0.25),
        ]
        candidates: list[dict] = []
        for entry_threshold, full_position_threshold in threshold_pairs:
            for max_position in [0.15, 0.25, 0.35]:
                for min_holding_hours in [48, 72]:
                    for drawdown_de_risk_threshold, de_risk_position in risk_pairs:
                        candidates.append(
                            {
                                "entry_threshold": entry_threshold,
                                "exit_threshold": 0.0,
                                "full_position_threshold": full_position_threshold,
                                "max_position": max_position,
                                "min_holding_hours": min_holding_hours,
                                "cooldown_hours": 12,
                                "drawdown_de_risk_threshold": drawdown_de_risk_threshold,
                                "de_risk_position": de_risk_position,
                            }
                        )
        return candidates
    raise ValueError(f"unknown scan profile: {profile}")


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


def select_top_feasible_candidates(
    frame: pd.DataFrame,
    *,
    limit: int = 5,
    min_annualized_return: float = 0.04,
    max_drawdown: float = 0.10,
) -> pd.DataFrame:
    ranked = rank_scan_results(
        frame,
        min_annualized_return=min_annualized_return,
        max_drawdown=max_drawdown,
    )
    feasible = ranked[ranked["meets_constraints"]].copy()
    return feasible.head(limit).reset_index(drop=True)


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
