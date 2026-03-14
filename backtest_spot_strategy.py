from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import qlib
from qlib.backtest import backtest as qlib_backtest
from qlib.constant import REG_CN

from strategy_config import SpotStrategyConfig


REQUIRED_COLUMNS = ["pred_return"]


def normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame missing required columns: {missing}")

    normalized = frame.copy()
    if isinstance(normalized.index, pd.MultiIndex):
        normalized.index = pd.to_datetime(normalized.index.get_level_values(0))
    else:
        normalized.index = pd.to_datetime(normalized.index)

    normalized = normalized.sort_index()
    if normalized.index.has_duplicates:
        raise ValueError("prediction frame contains duplicate timestamps after normalization")
    return normalized


def load_prediction_frames(paths: Iterable[Path]) -> pd.DataFrame:
    normalized_frames = [normalize_prediction_frame(pd.read_pickle(path)) for path in paths]
    if not normalized_frames:
        raise ValueError("no prediction files were provided")

    combined = pd.concat(normalized_frames).sort_index()
    if combined.index.has_duplicates:
        raise ValueError("combined prediction frames contain duplicate timestamps")
    return combined


def prepare_signal_frame(frame: pd.DataFrame, instrument: str) -> pd.DataFrame:
    signal = frame.loc[:, ["pred_return"]].rename(columns={"pred_return": "score"}).copy()
    signal["instrument"] = instrument
    signal.index.name = "datetime"
    signal = signal.reset_index().set_index(["instrument", "datetime"]).sort_index()
    return signal


def build_zero_benchmark(signal: pd.DataFrame) -> pd.Series:
    datetimes = signal.index.get_level_values("datetime").unique().sort_values()
    return pd.Series(0.0, index=datetimes)


def build_backtest_components(
    signal: pd.DataFrame,
    config: SpotStrategyConfig,
) -> tuple[dict, dict, dict]:
    benchmark = build_zero_benchmark(signal)
    strategy_config = {
        "class": "QlibLongFlatStrategy",
        "module_path": "qlib_spot_strategy",
        "kwargs": {
            "signal": signal,
            "instrument": config.instrument,
            "time_per_step": config.freq,
            "entry_threshold": config.entry_threshold,
            "exit_threshold": config.exit_threshold,
            "full_position_threshold": config.full_position_threshold,
            "min_holding_hours": config.min_holding_hours,
            "cooldown_hours": config.cooldown_hours,
            "max_position": config.max_position,
            "drawdown_de_risk_threshold": config.drawdown_de_risk_threshold,
            "de_risk_position": config.de_risk_position,
            "risk_degree": 1.0,
        },
    }
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": config.freq,
            "generate_portfolio_metrics": True,
        },
    }
    backtest_config = {
        "start_time": config.start_time,
        "end_time": config.end_time,
        "benchmark": benchmark,
        "account": config.initial_cash,
        "exchange_kwargs": {
            "freq": config.freq,
            "codes": [config.instrument],
            "deal_price": config.deal_price,
            "open_cost": config.fee_rate,
            "close_cost": config.fee_rate,
            "min_cost": config.min_cost,
            "trade_unit": None,
            "limit_threshold": None,
        },
        "pos_type": "Position",
    }
    return strategy_config, executor_config, backtest_config


def run_qlib_backtest(signal: pd.DataFrame, config: SpotStrategyConfig) -> tuple[pd.DataFrame, object, pd.DataFrame | None]:
    qlib.init(provider_uri=config.provider_uri, region=REG_CN)
    strategy_config, executor_config, backtest_config = build_backtest_components(signal, config)
    portfolio_metric_dict, indicator_dict = qlib_backtest(
        strategy=strategy_config,
        executor=executor_config,
        **backtest_config,
    )

    report = None
    positions = None
    for _, (freq_report, freq_positions) in portfolio_metric_dict.items():
        report = freq_report
        positions = freq_positions
        break

    indicators = None
    if indicator_dict:
        for _, indicator_value in indicator_dict.items():
            indicators = indicator_value[0]
            break

    if report is None:
        raise RuntimeError("qlib backtest did not return a portfolio report")
    return report, positions, indicators


def summarize_report(report: pd.DataFrame, annualization_periods: int = 24 * 365) -> dict[str, float | None]:
    if report.empty:
        raise ValueError("cannot summarize an empty report")

    net_returns = (report["return"] - report["cost"]).astype(float)
    cumulative_curve = (1.0 + net_returns).cumprod()
    total_return = float(cumulative_curve.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (annualization_periods / len(net_returns)) - 1.0)
    annualized_volatility = float(net_returns.std(ddof=0) * (annualization_periods**0.5))
    max_drawdown = float((cumulative_curve / cumulative_curve.cummax() - 1.0).min())
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    exposure_ratio = None
    avg_holding_hours = None
    if "value" in report.columns and "account" in report.columns:
        exposure = (report["value"] / report["account"]).fillna(0.0).astype(float)
        exposure_ratio = float(exposure.mean())

        bar_hours = 1.0
        if len(report.index) > 1:
            deltas = report.index.to_series().diff().dropna()
            if not deltas.empty:
                bar_hours = deltas.dt.total_seconds().median() / 3600.0

        holding_hours: list[float] = []
        current_holding = 0.0
        for exposure_value in exposure:
            if exposure_value > 0:
                current_holding += bar_hours
            elif current_holding > 0:
                holding_hours.append(current_holding)
                current_holding = 0.0
        if current_holding > 0:
            holding_hours.append(current_holding)
        avg_holding_hours = (
            float(sum(holding_hours) / len(holding_hours)) if holding_hours else 0.0
        )

    summary: dict[str, float | None] = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": abs(max_drawdown),
        "total_cost": float(report["cost"].sum()),
        "mean_excess_return": float(net_returns.mean()),
        "turnover": float(report["turnover"].sum()) if "turnover" in report.columns else None,
        "exposure_ratio": exposure_ratio,
        "avg_holding_hours": avg_holding_hours,
    }
    return summary


def compute_monthly_returns(report: pd.DataFrame) -> pd.DataFrame:
    net_returns = (report["return"] - report["cost"]).astype(float)
    monthly = (
        net_returns.groupby(net_returns.index.to_period("M"))
        .apply(lambda series: (1.0 + series).prod() - 1.0)
        .to_frame("monthly_return")
    )
    monthly.index = monthly.index.astype(str)
    return monthly


def _expand_prediction_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(match) for match in sorted(glob.glob(pattern)))

    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise ValueError("no prediction files matched the provided glob patterns")
    return unique_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Qlib-first fee-aware BTCUSDT spot long/flat backtest on prediction files."
    )
    parser.add_argument(
        "--pred-glob",
        action="append",
        required=True,
        help="Glob pattern for prediction pickle files. Repeat the flag for multiple patterns.",
    )
    parser.add_argument("--provider-uri", default="./qlib_data/my_crypto_data")
    parser.add_argument("--instrument", default="BTCUSDT")
    parser.add_argument("--start-time", default="2024-01-01 00:00:00")
    parser.add_argument("--end-time", default="2024-12-31 23:00:00")
    parser.add_argument("--freq", default="60min")
    parser.add_argument(
        "--output-dir",
        default="reports/spot_backtest",
        help="Directory where raw report, monthly returns, summary JSON, and positions are written.",
    )
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--min-cost", type=float, default=0.0)
    parser.add_argument("--deal-price", default="close")
    parser.add_argument("--entry-threshold", type=float, default=0.0025)
    parser.add_argument("--exit-threshold", type=float, default=0.0005)
    parser.add_argument("--full-position-threshold", type=float, default=0.005)
    parser.add_argument("--min-holding-hours", type=int, default=24)
    parser.add_argument("--cooldown-hours", type=int, default=12)
    parser.add_argument("--max-position", type=float, default=1.0)
    parser.add_argument("--drawdown-de-risk-threshold", type=float, default=0.08)
    parser.add_argument("--de-risk-position", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SpotStrategyConfig(
        provider_uri=args.provider_uri,
        instrument=args.instrument,
        start_time=args.start_time,
        end_time=args.end_time,
        freq=args.freq,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        min_cost=args.min_cost,
        deal_price=args.deal_price,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        full_position_threshold=args.full_position_threshold,
        min_holding_hours=args.min_holding_hours,
        cooldown_hours=args.cooldown_hours,
        max_position=args.max_position,
        drawdown_de_risk_threshold=args.drawdown_de_risk_threshold,
        de_risk_position=args.de_risk_position,
    )
    config.validate()

    prediction_paths = _expand_prediction_globs(args.pred_glob)
    combined = load_prediction_frames(prediction_paths)
    signal = prepare_signal_frame(combined, instrument=config.instrument)
    report, positions, indicators = run_qlib_backtest(signal, config)
    summary = summarize_report(report, annualization_periods=config.annualization_periods)
    monthly = compute_monthly_returns(report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "report.csv")
    monthly.to_csv(output_dir / "monthly_returns.csv")
    pd.to_pickle(positions, output_dir / "positions.pkl")
    if indicators is not None:
        indicators.to_csv(output_dir / "indicators.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
