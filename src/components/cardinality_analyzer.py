"""
Enterprise Cardinality Analyzer
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.logger.logger import logger


class CardinalityAnalyzer:
    """
    Enterprise Cardinality Analyzer

    Features
    --------
    ✓ Detect High Cardinality Columns
    ✓ Remove High Cardinality Columns
    ✓ Generate Report
    """

    def __init__(self):

        self.report = {
            "high_cardinality_columns": {},
            "removed_columns": [],
        }

    def clean(
        self,
        dataframe: pd.DataFrame,
        threshold: float = 0.90,
    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Cardinality Analysis Started")
        logger.info("=" * 70)

        df = dataframe.copy()

        self.generate_report(df)

        df = self.remove_high_cardinality(
            df,
            threshold,
        )

        logger.info("=" * 70)
        logger.info("Cardinality Analysis Completed")
        logger.info("=" * 70)

        return df


    def generate_report(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        report = {}

        categorical_columns = dataframe.select_dtypes(
            include=["object", "category"]
        ).columns

        for column in categorical_columns:

            unique = dataframe[column].nunique(dropna=False)

            ratio = round(
                unique / len(dataframe),
                4,
            )

            report[column] = {
                "unique_values": unique,
                "unique_ratio": ratio,
            }

        self.report[
            "high_cardinality_columns"
        ] = report

        logger.info(
            f"Categorical Columns Analysed : {len(report)}"
        )
    def remove_high_cardinality(
        self,
        dataframe: pd.DataFrame,
        threshold: float = 0.90,
    ) -> pd.DataFrame:

        remove_columns = []

        categorical_columns = dataframe.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns

        for column in categorical_columns:

            ratio = (
                dataframe[column].nunique(dropna=False)
                / len(dataframe)
            )

            if ratio >= threshold:

                remove_columns.append(column)

        if remove_columns:

            dataframe = dataframe.drop(
                columns=remove_columns
            )

        self.report[
            "removed_columns"
        ] = remove_columns

        logger.info(
            f"High Cardinality Columns Removed : {len(remove_columns)}"
        )

        return dataframe


    def statistics(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        rows = []

        categorical_columns = dataframe.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns

        for column in categorical_columns:

            unique = dataframe[column].nunique(
                dropna=False
            )

            ratio = round(
                unique / len(dataframe),
                4,
            )

            rows.append(
                {
                    "Column": column,
                    "Unique Values": unique,
                    "Unique Ratio": ratio,
                }
            )

        stats = pd.DataFrame(rows)

        if not stats.empty:

            stats = stats.sort_values(
                by="Unique Ratio",
                ascending=False,
            ).reset_index(drop=True)

        return stats    
    
    def summary(
        self,
    ) -> Dict:

        logger.info("=" * 70)
        logger.info("CARDINALITY ANALYZER REPORT")
        logger.info("=" * 70)

        for key, value in self.report.items():

            logger.info(f"{key} : {value}")

        logger.info("=" * 70)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/cardinality_report.csv",
    ) -> None:

        import os

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True,
        )

        rows = []

        for key, value in self.report.items():

            rows.append(
                {
                    "Metric": key,
                    "Value": str(value),
                }
            )

        pd.DataFrame(rows).to_csv(
            output_path,
            index=False,
        )

        logger.info(
            f"Cardinality report saved to {output_path}"
        )


    def get_high_cardinality_columns(
        self,
        dataframe: pd.DataFrame,
        threshold: float = 0.90,
    ) -> list[str]:

        columns = []

        categorical_columns = dataframe.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns

        for column in categorical_columns:

            ratio = (
                dataframe[column].nunique(dropna=False)
                / len(dataframe)
            )

            if ratio >= threshold:

                columns.append(column)

        return columns