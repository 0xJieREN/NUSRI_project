from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import tomllib

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


def build_probability_parameter_grid(
    *,
    enter_prob_thresholds: list[float],
    exit_prob_thresholds: list[float],
    full_prob_thresholds: list[float],
    max_positions: list[float],
    min_holding_hours_list: list[int],
    cooldown_hours_list: list[int],
    drawdown_thresholds: list[float],
    de_risk_positions: list[float],
) -> list[dict]:
    candidates: list[dict] = []
    for values in product(
        enter_prob_thresholds,
        exit_prob_thresholds,
        full_prob_thresholds,
        max_positions,
        min_holding_hours_list,
        cooldown_hours_list,
        drawdown_thresholds,
        de_risk_positions,
    ):
        (
            enter_prob_threshold,
            exit_prob_threshold,
            full_prob_threshold,
            max_position,
            min_holding_hours,
            cooldown_hours,
            drawdown_de_risk_threshold,
            de_risk_position,
        ) = values

        if full_prob_threshold < enter_prob_threshold:
            continue
        if enter_prob_threshold < exit_prob_threshold:
            continue
        if de_risk_position > max_position:
            continue

        candidates.append(
            {
                "enter_prob_threshold": enter_prob_threshold,
                "exit_prob_threshold": exit_prob_threshold,
                "full_prob_threshold": full_prob_threshold,
                "max_position": max_position,
                "min_holding_hours": min_holding_hours,
                "cooldown_hours": cooldown_hours,
                "drawdown_de_risk_threshold": drawdown_de_risk_threshold,
                "de_risk_position": de_risk_position,
            }
        )
    return candidates


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config.toml"


def _load_scan_profile_definition(profile: str, config_path: str | Path | None = None) -> dict:
    path = _default_config_path() if config_path is None else Path(config_path)
    with path.open("rb") as file_obj:
        config = tomllib.load(file_obj)
    profile_table = config.get("scan_profiles", {}).get(profile)
    if not isinstance(profile_table, dict):
        raise ValueError(f"unknown scan profile: {profile}")
    return profile_table


def _build_paired_scan_profile(raw: dict) -> list[dict]:
    threshold_pairs = [(float(pair[0]), float(pair[1])) for pair in raw["threshold_pairs"]]
    risk_pairs = [(float(pair[0]), float(pair[1])) for pair in raw["risk_pairs"]]
    max_positions = [float(value) for value in raw["max_positions"]]
    min_holding_hours_list = [int(value) for value in raw["min_holding_hours_list"]]
    cooldown_hours = int(raw["cooldown_hours"])

    candidates: list[dict] = []
    for entry_threshold, full_position_threshold in threshold_pairs:
        for max_position in max_positions:
            for min_holding_hours in min_holding_hours_list:
                for drawdown_de_risk_threshold, de_risk_position in risk_pairs:
                    candidates.append(
                        {
                            "entry_threshold": entry_threshold,
                            "exit_threshold": 0.0,
                            "full_position_threshold": full_position_threshold,
                            "max_position": max_position,
                            "min_holding_hours": min_holding_hours,
                            "cooldown_hours": cooldown_hours,
                            "drawdown_de_risk_threshold": drawdown_de_risk_threshold,
                            "de_risk_position": de_risk_position,
                        }
                    )
    return candidates


def build_scan_profile(profile: str, *, config_path: str | Path | None = None) -> list[dict]:
    raw = _load_scan_profile_definition(profile, config_path=config_path)
    kind = str(raw["kind"])
    if kind == "grid":
        return build_parameter_grid(
            entry_thresholds=[float(value) for value in raw["entry_thresholds"]],
            exit_thresholds=[float(value) for value in raw["exit_thresholds"]],
            full_position_thresholds=[float(value) for value in raw["full_position_thresholds"]],
            max_positions=[float(value) for value in raw.get("max_positions", [1.0])],
            min_holding_hours_list=[int(value) for value in raw["min_holding_hours_list"]],
            cooldown_hours_list=[int(value) for value in raw["cooldown_hours_list"]],
            drawdown_thresholds=[float(value) for value in raw["drawdown_thresholds"]],
            de_risk_positions=[float(value) for value in raw["de_risk_positions"]],
        )
    if kind == "probability_grid":
        return build_probability_parameter_grid(
            enter_prob_thresholds=[float(value) for value in raw["enter_prob_thresholds"]],
            exit_prob_thresholds=[float(value) for value in raw["exit_prob_thresholds"]],
            full_prob_thresholds=[float(value) for value in raw["full_prob_thresholds"]],
            max_positions=[float(value) for value in raw["max_positions"]],
            min_holding_hours_list=[int(value) for value in raw["min_holding_hours_list"]],
            cooldown_hours_list=[int(value) for value in raw["cooldown_hours_list"]],
            drawdown_thresholds=[float(value) for value in raw["drawdown_thresholds"]],
            de_risk_positions=[float(value) for value in raw["de_risk_positions"]],
        )
    if kind == "paired":
        return _build_paired_scan_profile(raw)
    raise ValueError(f"unsupported scan profile kind: {kind}")


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
    signal_column = {
        "probability": "pred_prob",
        "return": "pred_return",
        "score": "pred_score",
    }[config.signal_kind]
    combined = load_prediction_frames(prediction_files, signal_column=signal_column)
    signal = prepare_signal_frame(combined, instrument=config.instrument, signal_column=signal_column)
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
