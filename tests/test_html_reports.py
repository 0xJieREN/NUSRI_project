from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pandas as pd

from nusri_project.reporting.html_reports import (
    detect_experiment_layout,
    generate_experiment_report,
)


class HtmlReportsTests(unittest.TestCase):
    def test_detect_experiment_layout_for_single_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "report.csv").write_text("datetime,account,return,total_turnover,turnover,total_cost,cost,value,cash,bench\n")
            layout = detect_experiment_layout(root)

        self.assertEqual(layout, "single_run")

    def test_detect_experiment_layout_for_summary_only(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.csv").write_text("annualized_return,sharpe\n0.1,1.2\n")
            layout = detect_experiment_layout(root)

        self.assertEqual(layout, "summary_only")

    def test_generate_experiment_report_creates_html_for_single_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = Path(tmp) / "html"
            report = pd.DataFrame(
                {
                    "datetime": pd.date_range("2024-01-01", periods=4, freq="h"),
                    "account": [100000, 101000, 100500, 102000],
                    "return": [0.0, 0.01, -0.0049505, 0.014925],
                    "total_turnover": [0.0, 1.0, 1.5, 2.0],
                    "turnover": [0.0, 1.0, 0.5, 0.5],
                    "total_cost": [0.0, 100.0, 150.0, 200.0],
                    "cost": [0.0, 0.001, 0.0005, 0.0005],
                    "value": [0.0, 50000.0, 45000.0, 55000.0],
                    "cash": [100000.0, 50000.0, 55500.0, 47000.0],
                    "bench": [0.0, 0.0, 0.0, 0.0],
                }
            )
            report.to_csv(root / "report.csv", index=False)
            monthly = pd.DataFrame({"datetime": ["2024-01"], "monthly_return": [0.02]})
            monthly.to_csv(root / "monthly_returns.csv", index=False)
            (root / "summary.json").write_text(json.dumps({"annualized_return": 0.12, "sharpe": 1.1}))

            html_path = generate_experiment_report(root, out)

            self.assertTrue(html_path.exists())
            self.assertIn("收益曲线", html_path.read_text())

    def test_generate_experiment_report_creates_html_for_summary_only(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = Path(tmp) / "html"
            summary = pd.DataFrame(
                [
                    {"label_horizon_hours": 24, "shell_name": "balanced", "annualized_return": 0.1, "sharpe": 1.0},
                    {"label_horizon_hours": 72, "shell_name": "balanced", "annualized_return": 0.2, "sharpe": 1.4},
                ]
            )
            summary.to_csv(root / "summary.csv", index=False)

            html_path = generate_experiment_report(root, out)

            self.assertTrue(html_path.exists())
            self.assertIn("实验汇总", html_path.read_text())


if __name__ == "__main__":
    unittest.main()
