from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd

from nusri_project.config.runtime_config import load_runtime_config
from nusri_project.strategy.strategy_config import (
    SpotStrategyConfig,
    build_spot_strategy_config_from_runtime,
)
from scripts.analysis.generate_html_reports import update_html_reports
from nusri_project.strategy.phase2_strategy_research import (
    build_scan_profile,
    rank_scan_results,
    run_parameter_scan,
    select_top_feasible_candidates,
)
from nusri_project.strategy.research_profiles import find_horizon_prediction_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted trading-layer tuning for the 72h label.")
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--provider-uri", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment-profile", default=None)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--instrument", default="BTCUSDT")
    parser.add_argument("--freq", default="60min")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--min-cost", type=float, default=0.0)
    parser.add_argument("--deal-price", default="close")
    parser.add_argument("--scan-profile", default="label72_trade_tuning_fast")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--update-html", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    args = parse_args()
    prediction_files = find_horizon_prediction_files(
        Path(args.predictions_root) / "72h",
        label_horizon_hours=72,
        year=args.year,
    )
    if not prediction_files:
        raise FileNotFoundError(f"no 72h prediction files found for {args.year} under {args.predictions_root}")

    if args.config is not None:
        runtime = load_runtime_config(args.config, experiment_name=args.experiment_profile)
        config = build_spot_strategy_config_from_runtime(
            runtime,
            start_time=f"{args.year}-01-01 00:00:00",
            end_time=f"{args.year}-12-31 23:00:00",
        )
    else:
        if args.provider_uri is None:
            raise ValueError("--provider-uri is required when --config is not provided")
        config = SpotStrategyConfig(
            provider_uri=args.provider_uri,
            instrument=args.instrument,
            start_time=f"{args.year}-01-01 00:00:00",
            end_time=f"{args.year}-12-31 23:00:00",
            freq=args.freq,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            min_cost=args.min_cost,
            deal_price=args.deal_price,
        )
    config.validate()

    output_dir = Path(args.output_dir or f"reports/tune-72h-{args.year}")
    scan_results = run_parameter_scan(
        prediction_files,
        config,
        build_scan_profile(args.scan_profile, config_path=args.config),
    )
    ranked = rank_scan_results(scan_results)
    top_candidates = select_top_feasible_candidates(scan_results, limit=args.top_k)

    output_dir.mkdir(parents=True, exist_ok=True)
    scan_results.to_csv(output_dir / f"scan_results_{args.year}.csv", index=False)
    ranked.to_csv(output_dir / f"scan_ranked_{args.year}.csv", index=False)
    top_candidates.to_csv(output_dir / f"top_candidates_{args.year}.csv", index=False)

    print(ranked.head(15).to_string(index=False))
    print("\nTop feasible:")
    print(top_candidates.to_string(index=False))

    if args.update_html:
        update_html_reports(
            reports_root=Path("reports"),
            output_root=Path("reports/html"),
            experiments=[output_dir.name],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
