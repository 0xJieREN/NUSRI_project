from __future__ import annotations

import argparse
from pathlib import Path
import warnings

from nusri_project.config.runtime_config import load_runtime_config
from nusri_project.strategy.strategy_config import (
    SpotStrategyConfig,
    build_spot_strategy_config_from_runtime,
)
from scripts.analysis.generate_html_reports import update_html_reports
from nusri_project.strategy.phase2_strategy_research import (
    baseline_summary_row,
    build_scan_profile,
    find_prediction_files,
    rank_scan_results,
    run_parameter_scan,
    run_strategy_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 2024 BTCUSDT spot baseline and a small Qlib-first parameter scan."
    )
    parser.add_argument("--mlruns-root", required=True)
    parser.add_argument("--provider-uri", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment-profile", default=None)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--instrument", default="BTCUSDT")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--freq", default="60min")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--min-cost", type=float, default=0.0)
    parser.add_argument("--deal-price", default="close")
    parser.add_argument("--entry-threshold", type=float, default=0.0005)
    parser.add_argument("--exit-threshold", type=float, default=0.0)
    parser.add_argument("--full-position-threshold", type=float, default=0.001)
    parser.add_argument("--min-holding-hours", type=int, default=24)
    parser.add_argument("--cooldown-hours", type=int, default=12)
    parser.add_argument("--max-position", type=float, default=1.0)
    parser.add_argument("--drawdown-de-risk-threshold", type=float, default=0.08)
    parser.add_argument("--de-risk-position", type=float, default=0.5)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--scan-profile", default="small", choices=("small", "conservative", "conservative_fast"))
    parser.add_argument("--update-html", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    args = parse_args()
    prediction_files = find_prediction_files(Path(args.mlruns_root), year=args.year)
    if not prediction_files:
        raise FileNotFoundError(f"no prediction files found for {args.year} under {args.mlruns_root}")

    start_time = f"{args.year}-01-01 00:00:00"
    end_time = f"{args.year}-12-31 23:00:00"
    if args.config is not None:
        runtime = load_runtime_config(args.config, experiment_name=args.experiment_profile)
        config = build_spot_strategy_config_from_runtime(
            runtime,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        if args.provider_uri is None:
            raise ValueError("--provider-uri is required when --config is not provided")
        config = SpotStrategyConfig(
            provider_uri=args.provider_uri,
            instrument=args.instrument,
            start_time=start_time,
            end_time=end_time,
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

    output_dir = Path(args.output_dir or f"reports/baseline-{args.year}")
    baseline_summary = run_strategy_config(
        prediction_files,
        config,
        output_dir=output_dir / "baseline",
    )
    baseline_frame = (
        __import__("pandas").DataFrame([baseline_summary_row(config, baseline_summary)])
    )
    baseline_frame.to_csv(output_dir / "baseline_summary.csv", index=False)

    if args.scan:
        parameter_grid = build_scan_profile(args.scan_profile, config_path=args.config)
        scan_results = run_parameter_scan(prediction_files, config, parameter_grid)
        ranked_results = rank_scan_results(scan_results)
        scan_results.to_csv(output_dir / "scan_results.csv", index=False)
        ranked_results.to_csv(output_dir / "scan_ranked.csv", index=False)
        print(ranked_results.head(10).to_string(index=False))
    else:
        print(baseline_frame.to_string(index=False))

    if args.update_html:
        update_html_reports(
            reports_root=Path("reports"),
            output_root=Path("reports/html"),
            experiments=[output_dir.name],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
