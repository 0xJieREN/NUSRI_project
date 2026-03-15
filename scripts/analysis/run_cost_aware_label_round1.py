from __future__ import annotations

import argparse
from pathlib import Path

from nusri_project.strategy.cost_aware_label_round1 import (
    build_cost_aware_round1_matrix,
    build_cost_aware_round1_modes,
    evaluate_cost_aware_round1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cost-aware label round 1 comparison.")
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--output-root", default="reports/cost_aware_label_round1")
    parser.add_argument("--year", type=int, default=2024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("label_modes:", build_cost_aware_round1_modes())
    print("matrix_size:", len(build_cost_aware_round1_matrix()))
    summary = evaluate_cost_aware_round1(
        predictions_root=Path(args.predictions_root),
        output_root=Path(args.output_root),
        provider_uri=args.provider_uri,
        year=args.year,
    )
    if not summary.empty:
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_root / "summary.csv", index=False)
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
