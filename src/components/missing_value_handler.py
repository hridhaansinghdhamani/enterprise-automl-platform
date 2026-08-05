"""
Enterprise Missing Value Handler
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from sklearn.impute import SimpleImputer

from src.logger.logger import logger


class MissingValueHandler:
    """
    Enterprise Missing Value Handler

    Features
    --------
    ✓ Missing Value Report
    ✓ Automatic Numerical Imputation
    ✓ Automatic Categorical Imputation
    ✓ Threshold Based Column Removal
    """

    def __init__(self):

        self.report = {
            "columns_with_missing": {},
            "removed_columns": [],
            "numerical_imputed": [],
            "categorical_imputed": [],
        }

    def clean(
        self,
        dataframe: pd.DataFrame,
        threshold: float = 0.60,
    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Missing Value Handling Started")
        logger.info("=" * 70)

        df = dataframe.copy()

        self.generate_report(df)

        df = self.remove_high_missing_columns(
            df,
            threshold,
        )

        df = self.impute_numerical(df)

        df = self.impute_categorical(df)

        logger.info("=" * 70)
        logger.info("Missing Value Handling Completed")
        logger.info("=" * 70)

        return df


    def generate_report(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        report = {}

        for column in dataframe.columns:

            missing = int(
                dataframe[column].isna().sum()
            )

            if missing > 0:

                report[column] = {
                    "count": missing,
                    "percentage": round(
                        missing / len(dataframe) * 100,
                        2,
                    ),
                }

        self.report[
            "columns_with_missing"
        ] = report

        logger.info(
            f"Columns with Missing Values : {len(report)}"
        )

    def remove_high_missing_columns(
        self,
        dataframe: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:

        remove_columns = []

        for column in dataframe.columns:

            missing_ratio = (
                dataframe[column].isna().sum()
                / len(dataframe)
            )

            if missing_ratio >= threshold:

                remove_columns.append(column)

        if remove_columns:

            dataframe = dataframe.drop(
                columns=remove_columns
            )

        self.report[
            "removed_columns"
        ] = remove_columns

        logger.info(
            f"Columns Removed (> {int(threshold*100)}% Missing): {len(remove_columns)}"
        )

        return dataframe


    def impute_numerical(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        numerical_columns = dataframe.select_dtypes(
            include=[
                "int16",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64",
            ]
        ).columns.tolist()

        if not numerical_columns:

            return dataframe

        imputer = SimpleImputer(
            strategy="median"
        )

        dataframe[numerical_columns] = imputer.fit_transform(
            dataframe[numerical_columns]
        )

        self.report[
            "numerical_imputed"
        ] = numerical_columns

        logger.info(
            f"Numerical Columns Imputed : {len(numerical_columns)}"
        )

        return dataframe


    def impute_categorical(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        categorical_columns = dataframe.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns.tolist()

        if not categorical_columns:

            return dataframe

        imputer = SimpleImputer(
            strategy="most_frequent"
        )

        dataframe[categorical_columns] = imputer.fit_transform(
            dataframe[categorical_columns]
        )

        self.report[
            "categorical_imputed"
        ] = categorical_columns

        logger.info(
            f"Categorical Columns Imputed : {len(categorical_columns)}"
        )

        return dataframe    
    
    def summary(
        self,
    ) -> Dict:

        logger.info("=" * 70)
        logger.info("MISSING VALUE HANDLER REPORT")
        logger.info("=" * 70)

        for key, value in self.report.items():

            logger.info(f"{key} : {value}")

        logger.info("=" * 70)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/missing_value_report.csv",
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
            f"Missing value report saved to {output_path}"
        )


    def statistics(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        rows = []

        for column in dataframe.columns:

            missing = int(
                dataframe[column].isna().sum()
            )

            percentage = round(
                (missing / len(dataframe)) * 100,
                2,
            )

            rows.append(
                {
                    "Column": column,
                    "Missing Count": missing,
                    "Missing Percentage": percentage,
                }
            )

        stats = pd.DataFrame(rows)

        return stats.sort_values(
            by="Missing Percentage",
            ascending=False,
        ).reset_index(drop=True)