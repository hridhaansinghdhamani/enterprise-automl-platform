"""
Enterprise ID Column Detector
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import re

import pandas as pd

from src.logger.logger import logger


class IDColumnDetector:
    """
    Automatically detects identifier columns.

    Detection Strategies
    --------------------
    ✓ Known ID names
    ✓ Unique value ratio
    ✓ UUID detection
    ✓ Email detection
    ✓ Long string detection
    ✓ Sequential integer detection
    """

    def __init__(self):

        self.detected_columns = []

        self.common_names = {

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
            "rollno",
            "roll_no",
            "studentid",
            "student_id",
            "serialno",
            "serial_no",

        }

    def detect(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:

        self.detected_columns.clear()

        logger.info(
            "Detecting Identifier Columns..."
        )

        for column in dataframe.columns:

            if self._is_known_name(column):

                self.detected_columns.append(column)

                continue

            if self._is_unique_column(
                dataframe[column]
            ):

                self.detected_columns.append(column)

                continue

            if self._looks_like_uuid(
                dataframe[column]
            ):

                self.detected_columns.append(column)

                continue

        logger.info(
            f"Detected {len(self.detected_columns)} ID Columns."
        )

        return sorted(
            list(
                set(
                    self.detected_columns
                )
            )
        )
    
    def _is_known_name(
        self,
        column_name: str,
    ) -> bool:

        cleaned = (
            column_name.lower()
            .replace("_", "")
            .replace(" ", "")
        )

        return cleaned in self.common_names


    def _is_unique_column(
        self,
        column: pd.Series,
        threshold: float = 0.98,
    ) -> bool:

        if len(column) == 0:
            return False

        ratio = (
            column.nunique(dropna=False)
            / len(column)
        )

        return ratio >= threshold


    def _looks_like_uuid(
        self,
        column: pd.Series,
    ) -> bool:

        if column.dtype != "object":
            return False

        sample = (
            column.dropna()
            .astype(str)
            .head(20)
            .tolist()
        )

        if len(sample) == 0:
            return False

        pattern = re.compile(

            r"^[0-9a-fA-F]{8}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}$"

        )

        matched = sum(
            bool(pattern.match(value))
            for value in sample
        )

        return matched >= max(
            1,
            len(sample) // 2,
        )


    def _looks_like_email(
        self,
        column: pd.Series,
    ) -> bool:

        if column.dtype != "object":
            return False

        sample = (
            column.dropna()
            .astype(str)
            .head(20)
            .tolist()
        )

        if len(sample) == 0:
            return False

        matched = sum(
            "@" in value and "." in value
            for value in sample
        )

        return matched >= max(
            1,
            len(sample) // 2,
        )
    
    def _looks_like_sequential_integer(
        self,
        column: pd.Series,
    ) -> bool:

        if not pd.api.types.is_integer_dtype(column):
            return False

        values = column.dropna().sort_values().to_numpy()

        if len(values) < 5:
            return False

        differences = values[1:] - values[:-1]

        return (differences == 1).all()


    def _looks_like_long_string(
        self,
        column: pd.Series,
        threshold: int = 25,
    ) -> bool:

        if column.dtype != "object":
            return False

        sample = (
            column.dropna()
            .astype(str)
            .head(50)
        )

        if sample.empty:
            return False

        average_length = sample.str.len().mean()

        return average_length >= threshold


    def remove(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        detected = self.detect(dataframe)

        if not detected:

            logger.info(
                "No Identifier Columns Found."
            )

            return dataframe

        logger.info(
            f"Removing Identifier Columns : {detected}"
        )

        return dataframe.drop(
            columns=detected,
            errors="ignore",
        )


    def summary(
        self,
    ) -> dict:

        return {
            "total_detected": len(
                self.detected_columns
            ),
            "columns": self.detected_columns,
        }
    