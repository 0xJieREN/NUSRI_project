from __future__ import annotations

import argparse
from pathlib import Path

from nusri_project.reporting.html_reports import generate_experiment_report


DEFAULT_EXPERIMENTS = [
    "phase2_2024_round2_fast_fixed",
    "phase2_2025_candidates",
    "label_optimization_round1",
    "cost_aware_label_round1_2025",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML experiment reports.")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--output-root", default="reports/html")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports_root = Path(args.reports_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    generated = []
    missing = []
    for name in args.experiments:
        experiment_dir = reports_root / name
        if not experiment_dir.exists():
            missing.append(name)
            continue
        html_path = generate_experiment_report(experiment_dir, output_root / name)
        generated.append((name, html_path))

    index_lines = [
        "<html><head><meta charset='utf-8'><title>实验报告索引</title></head><body>",
        "<h1>实验报告索引</h1>",
        "<ul>",
    ]
    for name, html_path in generated:
        rel = html_path.relative_to(output_root)
        index_lines.append(f"<li><a href='{rel.as_posix()}'>{name}</a></li>")
    index_lines.append("</ul>")
    if missing:
        index_lines.append("<h2>未找到的实验</h2><ul>")
        for name in missing:
            index_lines.append(f"<li>{name}</li>")
        index_lines.append("</ul>")
    index_lines.append("</body></html>")
    (output_root / "index.html").write_text("\n".join(index_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
