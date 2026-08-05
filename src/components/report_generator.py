import logging
import os
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates a PDF report for AutoML training results.
    """

    def generate(
        self,
        leaderboard: pd.DataFrame,
        output_path: str,
    ):
        """
        Generate a PDF report.

        Parameters
        ----------
        leaderboard : pd.DataFrame
            Model leaderboard sorted by performance.
        output_path : str
            Path to save the PDF.
        """

        try:
            logger.info("Generating AutoML report...")

            os.makedirs(
                os.path.dirname(output_path) or ".",
                exist_ok=True,
            )

            styles = getSampleStyleSheet()

            pdf = SimpleDocTemplate(output_path)

            story = []

            # -------------------------
            # Title
            # -------------------------
            story.append(
                Paragraph(
                    "Enterprise AutoML Report",
                    styles["Title"],
                )
            )

            story.append(Spacer(1, 20))

            # -------------------------
            # Generation Time
            # -------------------------
            story.append(
                Paragraph(
                    f"<b>Generated On:</b> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
                    styles["Normal"],
                )
            )

            story.append(Spacer(1, 15))

            # -------------------------
            # Summary
            # -------------------------
            story.append(
                Paragraph(
                    f"<b>Total Models Trained:</b> {len(leaderboard)}",
                    styles["Normal"],
                )
            )

            story.append(Spacer(1, 15))

            # -------------------------
            # Best Model
            # -------------------------
            if not leaderboard.empty:
                best_model = leaderboard.iloc[0]

                story.append(
                    Paragraph(
                        "<b>Best Model</b>",
                        styles["Heading2"],
                    )
                )

                for column in leaderboard.columns:
                    story.append(
                        Paragraph(
                            f"{column}: {best_model[column]}",
                            styles["Normal"],
                        )
                    )

                story.append(Spacer(1, 20))

            # -------------------------
            # Leaderboard Table
            # -------------------------
            story.append(
                Paragraph(
                    "<b>Model Leaderboard</b>",
                    styles["Heading2"],
                )
            )

            table_data = [
                leaderboard.columns.tolist()
            ] + leaderboard.astype(str).values.tolist()

            table = Table(table_data)

            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ]
                )
            )

            story.append(table)

            pdf.build(story)

            logger.info(
                "AutoML report saved to %s",
                output_path,
            )

            return output_path

        except Exception as e:
            logger.exception(
                "Failed to generate report: %s",
                str(e),
            )
            return None