from __future__ import annotations

import argparse
from pathlib import Path

from nusri_project.config.runtime_config import load_runtime_config
from scripts.analysis.generate_html_reports import update_html_reports
from nusri_project.strategy.cost_aware_label_round1 import (
    build_cost_aware_round1_matrix,
    build_cost_aware_round1_modes,
    evaluate_cost_aware_round1,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cost-aware label round 1 comparison.")
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--provider-uri", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment-profile", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--update-html", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    provider_uri = args.provider_uri
    if args.config is not None:
        runtime = load_runtime_config(args.config, experiment_name=args.experiment_profile)
        provider_uri = runtime.data.provider_uri
    if provider_uri is None:
        raise ValueError("--provider-uri is required when --config is not provided")

    output_root = Path(args.output_root or f"reports/costaware-{args.year}")
    print("label_modes:", build_cost_aware_round1_modes())
    print("matrix_size:", len(build_cost_aware_round1_matrix()))
    summary = evaluate_cost_aware_round1(
        predictions_root=Path(args.predictions_root),
        output_root=output_root,
        provider_uri=provider_uri,
        year=args.year,
    )
    if not summary.empty:
        output_root.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_root / "summary.csv", index=False)
        print(summary.to_string(index=False))

    if args.update_html:
        update_html_reports(
            reports_root=Path("reports"),
            output_root=Path("reports/html"),
            experiments=[output_root.name],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
