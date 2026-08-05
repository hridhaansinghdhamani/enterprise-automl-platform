"""
Enterprise DataType Optimizer
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.logger.logger import logger


class DataTypeOptimizer:
    """
    Enterprise DataType Optimizer

    Features
    --------
    ✓ Integer Downcasting
    ✓ Float Downcasting
    ✓ Category Conversion
    ✓ Boolean Detection
    ✓ Memory Usage Report
    """

    def __init__(self):

        self.report = {
            "before_memory_mb": 0.0,
            "after_memory_mb": 0.0,
            "saved_memory_mb": 0.0,
            "converted_columns": {},
        }

    def optimize(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Datatype Optimization Started")
        logger.info("=" * 70)

        df = dataframe.copy()

        self.report["before_memory_mb"] = round(
            df.memory_usage(deep=True).sum() / 1024**2,
            2,
        )

        df = self.optimize_integer(df)

        df = self.optimize_float(df)

        df = self.optimize_boolean(df)

        df = self.optimize_category(df)

        self.report["after_memory_mb"] = round(
            df.memory_usage(deep=True).sum() / 1024**2,
            2,
        )

        self.report["saved_memory_mb"] = round(
            self.report["before_memory_mb"]
            - self.report["after_memory_mb"],
            2,
        )

        logger.info(
            f"Memory Reduced : {self.report['saved_memory_mb']} MB"
        )

        logger.info("=" * 70)
        logger.info("Datatype Optimization Completed")
        logger.info("=" * 70)

        return df


    def optimize_integer(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        integer_columns = dataframe.select_dtypes(
            include=["int64"]
        ).columns

        for column in integer_columns:

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                downcast="integer",
            )

            self.report["converted_columns"][column] = str(
                dataframe[column].dtype
            )

        return dataframe
    
    def optimize_float(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        float_columns = dataframe.select_dtypes(
            include=["float64"]
        ).columns

        for column in float_columns:

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                downcast="float",
            )

            self.report["converted_columns"][column] = str(
                dataframe[column].dtype
            )

        return dataframe


    def optimize_boolean(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        object_columns = dataframe.select_dtypes(
            include=["object"]
        ).columns

        for column in object_columns:

            unique = (
                dataframe[column]
                .dropna()
                .unique()
                .tolist()
            )

            unique = [
                str(x).lower()
                for x in unique
            ]

            boolean_values = {
                "true",
                "false",
                "yes",
                "no",
                "0",
                "1",
            }

            if len(unique) <= 2 and set(unique).issubset(
                boolean_values
            ):

                dataframe[column] = (
                    dataframe[column]
                    .astype("category")
                )

                self.report[
                    "converted_columns"
                ][column] = "category"

        return dataframe


    def optimize_category(
        self,
        dataframe: pd.DataFrame,
        threshold: float = 0.50,
    ) -> pd.DataFrame:

        object_columns = dataframe.select_dtypes(
            include=["object"]
        ).columns

        for column in object_columns:

            ratio = (
                dataframe[column].nunique(dropna=False)
                / len(dataframe)
            )

            if ratio <= threshold:

                dataframe[column] = dataframe[
                    column
                ].astype("category")

                self.report[
                    "converted_columns"
                ][column] = "category"

        return dataframe
    
    def summary(
        self,
    ) -> Dict:

        logger.info("=" * 70)
        logger.info("DATATYPE OPTIMIZER REPORT")
        logger.info("=" * 70)

        for key, value in self.report.items():

            logger.info(f"{key} : {value}")

        logger.info("=" * 70)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/datatype_optimizer_report.csv",
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
            f"Datatype optimization report saved to {output_path}"
        )


    def memory_statistics(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict:

        return {
            "rows": len(dataframe),
            "columns": dataframe.shape[1],
            "memory_mb": round(
                dataframe.memory_usage(
                    deep=True
                ).sum()
                / 1024**2,
                2,
            ),
            "dtypes": dataframe.dtypes.astype(str).to_dict(),
        }