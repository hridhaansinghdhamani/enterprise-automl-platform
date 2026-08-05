"""
Enterprise Data Drift Monitoring
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import os
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


class DriftMonitor:
    """
    Generate Evidently AI Data Drift reports.
    """

    def __init__(self, report_dir: str = "reports/drift") -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        reference_data,
        current_data,
        report_name: str = "data_drift_report.html",
    ) -> str:
        """
        Generate HTML drift report.

        Args:
            reference_data: Training dataframe
            current_data: Current inference dataframe
            report_name: HTML report name

        Returns:
            Path to generated report
        """

        report = Report(
            metrics=[
                DataDriftPreset(),
            ]
        )

        report.run(
            reference_data=reference_data,
            current_data=current_data,
        )

        output_path = self.report_dir / report_name

        report.save_html(str(output_path))

        return str(output_path)

    def generate_json(
        self,
        reference_data,
        current_data,
        report_name: str = "data_drift_report.json",
    ) -> str:
        """
        Generate JSON drift report.
        """

        report = Report(
            metrics=[
                DataDriftPreset(),
            ]
        )

        report.run(
            reference_data=reference_data,
            current_data=current_data,
        )

        output_path = self.report_dir / report_name

        report.save_json(str(output_path))

        return str(output_path)