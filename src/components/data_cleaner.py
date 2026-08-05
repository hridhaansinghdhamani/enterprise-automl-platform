"""
Enterprise Data Cleaner
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.logger.logger import logger


class DataCleaner:
    """
    Enterprise Data Cleaning Engine
    """

    def __init__(self):

        self.report = {
            "original_shape": None,
            "final_shape": None,
            "removed_rows": 0,
            "removed_columns": [],
            "duplicate_columns": [],
            "constant_columns": [],
            "id_columns": [],
            "high_cardinality_columns": [],
            "datetime_columns": [],
            "missing_value_summary": {},
        }

    def clean(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info("=" * 80)
        logger.info("Starting Enterprise Data Cleaning")
        logger.info("=" * 80)

        df = dataframe.copy()

        self.report["original_shape"] = df.shape

        logger.info(f"Original Shape : {df.shape}")

        df = self.remove_duplicate_rows(df)
        df = self.remove_duplicate_columns(df)
        df = self.remove_constant_columns(df)
        df = self.remove_identifier_columns(df)
        df = self.remove_high_cardinality_columns(df)
        df = self.detect_datetime_columns(df)

        self.missing_value_summary(df)

        df = self.optimize_memory(df)

        self.report["final_shape"] = df.shape

        logger.info(f"Final Shape : {df.shape}")

        logger.info("=" * 80)
        logger.info("Enterprise Data Cleaning Completed")
        logger.info("=" * 80)

        return df

    def remove_duplicate_rows(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        before = len(dataframe)

        dataframe = dataframe.drop_duplicates().reset_index(drop=True)

        removed = before - len(dataframe)

        self.report["removed_rows"] = removed

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

        duplicate_columns = list(dict.fromkeys(duplicate_columns))

        if duplicate_columns:

            dataframe = dataframe.drop(columns=duplicate_columns)

        self.report["duplicate_columns"] = duplicate_columns

        self.report["removed_columns"].extend(
            duplicate_columns
        )

        logger.info(
            f"Duplicate Columns Removed : {len(duplicate_columns)}"
        )

        return dataframe


    def remove_constant_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        constant_columns = []

        for column in dataframe.columns:

            if dataframe[column].nunique(dropna=False) <= 1:

                constant_columns.append(column)

        if constant_columns:

            dataframe = dataframe.drop(columns=constant_columns)

        self.report["constant_columns"] = constant_columns

        self.report["removed_columns"].extend(
            constant_columns
        )

        logger.info(
            f"Constant Columns Removed : {len(constant_columns)}"
        )

        return dataframe


    def remove_identifier_columns(
        self,
        dataframe: pd.DataFrame,
        threshold: float = 0.98,
    ) -> pd.DataFrame:

        id_columns = []

        common_names = {
            "id",
            "customerid",
            "customer_id",
            "userid",
            "user_id",
            "employeeid",
            "employee_id",
            "orderid",
            "order_id",
            "invoiceid",
            "invoice_id",
            "transactionid",
            "transaction_id",
            "uuid",
            "guid",
        }

        for column in dataframe.columns:

            cleaned_name = (
                column.lower()
                .replace("_", "")
                .replace(" ", "")
            )

            if cleaned_name in common_names:

                id_columns.append(column)

                continue

            unique_ratio = (
                dataframe[column].nunique(dropna=False)
                / len(dataframe)
            )

            if unique_ratio >= threshold:

                id_columns.append(column)

        id_columns = list(dict.fromkeys(id_columns))

        if id_columns:

            dataframe = dataframe.drop(columns=id_columns)

        self.report["id_columns"] = id_columns

        self.report["removed_columns"].extend(id_columns)

        logger.info(
            f"Identifier Columns Removed : {len(id_columns)}"
        )

        return dataframe
    
    def remove_high_cardinality_columns(
        self,
        dataframe: pd.DataFrame,
        threshold: float = 0.90,
    ) -> pd.DataFrame:

        high_cardinality_columns = []

        categorical_columns = dataframe.select_dtypes(
            include=["object", "category"]
        ).columns

        for column in categorical_columns:

            unique_ratio = (
                dataframe[column].nunique(dropna=False)
                / len(dataframe)
            )

            if unique_ratio >= threshold:

                high_cardinality_columns.append(column)

        if high_cardinality_columns:

            dataframe = dataframe.drop(
                columns=high_cardinality_columns
            )

        self.report[
            "high_cardinality_columns"
        ] = high_cardinality_columns

        self.report[
            "removed_columns"
        ].extend(high_cardinality_columns)

        logger.info(
            f"High Cardinality Columns Removed : {len(high_cardinality_columns)}"
        )

        return dataframe


    def detect_datetime_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        datetime_columns = []

        for column in dataframe.columns:

            if dataframe[column].dtype != "object":
                continue

            try:

                converted = pd.to_datetime(
                    dataframe[column],
                    errors="raise",
                )

                dataframe[f"{column}_year"] = converted.dt.year
                dataframe[f"{column}_month"] = converted.dt.month
                dataframe[f"{column}_day"] = converted.dt.day
                dataframe[f"{column}_weekday"] = converted.dt.weekday

                dataframe = dataframe.drop(columns=[column])

                datetime_columns.append(column)

            except Exception:

                pass

        self.report["datetime_columns"] = datetime_columns

        logger.info(
            f"Datetime Columns Processed : {len(datetime_columns)}"
        )

        return dataframe


    def optimize_memory(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        for column in dataframe.select_dtypes(
            include=["int64"]
        ).columns:

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                downcast="integer",
            )

        for column in dataframe.select_dtypes(
            include=["float64"]
        ).columns:

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                downcast="float",
            )

        logger.info(
            "Memory Optimization Completed."
        )

        return dataframe
    
    def missing_value_summary(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        summary = {}

        for column in dataframe.columns:

            missing = int(
                dataframe[column].isna().sum()
            )

            if missing > 0:

                summary[column] = {
                    "count": missing,
                    "percentage": round(
                        (missing / len(dataframe)) * 100,
                        2,
                    ),
                }

        self.report[
            "missing_value_summary"
        ] = summary

        logger.info(
            f"Columns With Missing Values : {len(summary)}"
        )


    def summary(
        self,
    ) -> Dict:

        logger.info("=" * 80)
        logger.info("DATA CLEANING REPORT")
        logger.info("=" * 80)

        for key, value in self.report.items():

            logger.info(f"{key} : {value}")

        logger.info("=" * 80)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/data_cleaning_report.csv",
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
            f"Cleaning report saved to {output_path}"
        )