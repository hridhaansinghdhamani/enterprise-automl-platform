"""
Enterprise Duplicate Handler
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import pandas as pd

from src.logger.logger import logger


class DuplicateHandler:
    """
    Enterprise Duplicate Detection & Removal
    """

    def __init__(self):

        self.report = {
            "duplicate_rows_removed": 0,
            "duplicate_columns_removed": [],
        }

    def clean(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Duplicate Cleaning Started")
        logger.info("=" * 70)

        df = dataframe.copy()

        df = self.remove_duplicate_rows(df)

        df = self.remove_duplicate_columns(df)

        logger.info("=" * 70)
        logger.info("Duplicate Cleaning Completed")
        logger.info("=" * 70)

        return df


    def remove_duplicate_rows(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        before = len(dataframe)

        dataframe = dataframe.drop_duplicates().reset_index(
            drop=True
        )

        removed = before - len(dataframe)

        self.report[
            "duplicate_rows_removed"
        ] = removed

        logger.info(
            f"Duplicate Rows Removed : {removed}"
        )

        return dataframe


    def remove_duplicate_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        duplicate_columns = []

        columns = dataframe.columns.tolist()

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                if dataframe.iloc[:, i].equals(
                    dataframe.iloc[:, j]
                ):

                    duplicate_columns.append(columns[j])

        duplicate_columns = list(
            dict.fromkeys(duplicate_columns)
        )

        if duplicate_columns:

            dataframe = dataframe.drop(
                columns=duplicate_columns
            )

        self.report[
            "duplicate_columns_removed"
        ] = duplicate_columns

        logger.info(
            f"Duplicate Columns Removed : {len(duplicate_columns)}"
        )

        return dataframe
    
    def get_duplicate_rows(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        duplicates = dataframe[
            dataframe.duplicated(
                keep=False,
            )
        ]

        logger.info(
            f"Duplicate Rows Found : {len(duplicates)}"
        )

        return duplicates


    def get_duplicate_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:

        duplicate_columns = []

        columns = dataframe.columns.tolist()

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                if dataframe.iloc[:, i].equals(
                    dataframe.iloc[:, j]
                ):

                    duplicate_columns.append(
                        columns[j]
                    )

        duplicate_columns = list(
            dict.fromkeys(
                duplicate_columns
            )
        )

        logger.info(
            f"Duplicate Columns Found : {len(duplicate_columns)}"
        )

        return duplicate_columns


    def summary(
        self,
    ) -> dict:

        logger.info("=" * 70)
        logger.info("DUPLICATE HANDLER REPORT")
        logger.info("=" * 70)

        for key, value in self.report.items():

            logger.info(f"{key} : {value}")

        logger.info("=" * 70)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/duplicate_report.csv",
    ) -> None:

        import os

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True,
        )

        report_df = pd.DataFrame(
            [
                {
                    "Metric": key,
                    "Value": str(value),
                }
                for key, value in self.report.items()
            ]
        )

        report_df.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            f"Duplicate report saved to {output_path}"
        )