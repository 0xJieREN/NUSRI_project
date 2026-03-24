from __future__ import annotations

from pathlib import Path


def build_return_trading_shells() -> dict[str, dict]:
    return {
        "balanced": {
            "entry_threshold": 0.0015,
            "exit_threshold": 0.0,
            "full_position_threshold": 0.003,
            "max_position": 0.25,
            "min_holding_hours": 48,
            "cooldown_hours": 12,
            "drawdown_de_risk_threshold": 0.02,
            "de_risk_position": 0.25,
        },
        "conservative": {
            "entry_threshold": 0.0015,
            "exit_threshold": 0.0,
            "full_position_threshold": 0.003,
            "max_position": 0.15,
            "min_holding_hours": 48,
            "cooldown_hours": 12,
            "drawdown_de_risk_threshold": 0.02,
            "de_risk_position": 0.25,
        },
    }


def build_probability_trading_shells() -> dict[str, dict]:
    return {
        "balanced": {
            "signal_kind": "probability",
            "enter_prob_threshold": 0.65,
            "exit_prob_threshold": 0.50,
            "full_prob_threshold": 0.80,
            "max_position": 0.25,
            "min_holding_hours": 48,
            "cooldown_hours": 12,
            "drawdown_de_risk_threshold": 0.02,
            "de_risk_position": 0.10,
        },
        "conservative": {
            "signal_kind": "probability",
            "enter_prob_threshold": 0.65,
            "exit_prob_threshold": 0.50,
            "full_prob_threshold": 0.80,
            "max_position": 0.15,
            "min_holding_hours": 48,
            "cooldown_hours": 12,
            "drawdown_de_risk_threshold": 0.02,
            "de_risk_position": 0.0,
        },
    }


def find_horizon_prediction_files(
    prediction_dir: Path,
    *,
    label_horizon_hours: int,
    year: int,
) -> list[Path]:
    if not prediction_dir.exists() and prediction_dir.name == f"{label_horizon_hours}h":
        prediction_dir = prediction_dir.parent
    pattern = f"pred_{label_horizon_hours}h_{year}[0-1][0-9].pkl"
    return sorted(prediction_dir.glob(pattern), key=lambda path: path.name)
