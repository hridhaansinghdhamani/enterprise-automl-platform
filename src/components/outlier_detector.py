"""
Enterprise Outlier Detector
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.logger.logger import logger


class OutlierDetector:
    """
    Enterprise Outlier Detection Engine

    Methods
    -------
    ✓ IQR
    ✓ Z-Score
    ✓ Isolation Forest
    ✓ Local Outlier Factor
    """

    def __init__(self):

        self.report = {
            "method": None,
            "rows_before": 0,
            "rows_after": 0,
            "removed_rows": 0,
            "outlier_percentage": 0,
        }

    def clean(

        self,

        dataframe: pd.DataFrame,

        method: str = "auto",

    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Outlier Detection Started")
        logger.info("=" * 70)

        df = dataframe.copy()

        self.report["rows_before"] = len(df)

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_columns) == 0:

            logger.info("No Numerical Columns Found")

            return df

        if method == "auto":

            if len(df) < 5000:

                method = "iqr"

            else:

                method = "isolation"

        self.report["method"] = method

        if method == "iqr":

            df = self._iqr(df, numeric_columns)

        elif method == "zscore":

            df = self._zscore(df, numeric_columns)

        elif method == "isolation":

            df = self._isolation(df, numeric_columns)

        elif method == "lof":

            df = self._lof(df, numeric_columns)

        self.report["rows_after"] = len(df)

        self.report["removed_rows"] = (

            self.report["rows_before"]

            - self.report["rows_after"]

        )

        self.report["outlier_percentage"] = round(

            self.report["removed_rows"]

            / self.report["rows_before"]

            * 100,

            2,

        )

        logger.info(
            f"Removed {self.report['removed_rows']} Outliers"
        )

        return df
    
    def _iqr(
        self,
        dataframe: pd.DataFrame,
        columns,
    ) -> pd.DataFrame:

        mask = pd.Series(
            True,
            index=dataframe.index,
        )

        for column in columns:

            q1 = dataframe[column].quantile(0.25)
            q3 = dataframe[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            mask &= dataframe[column].between(
                lower,
                upper,
            )

        return dataframe.loc[
            mask
        ].reset_index(drop=True)


    def _zscore(
        self,
        dataframe: pd.DataFrame,
        columns,
        threshold: float = 3.0,
    ) -> pd.DataFrame:

        scores = np.abs(
            zscore(
                dataframe[columns],
                nan_policy="omit",
            )
        )

        mask = (
            scores < threshold
        ).all(axis=1)

        return dataframe.loc[
            mask
        ].reset_index(drop=True)


    def _isolation(
        self,
        dataframe: pd.DataFrame,
        columns,
    ) -> pd.DataFrame:

        model = IsolationForest(

            contamination="auto",

            random_state=42,

            n_jobs=-1,

        )

        prediction = model.fit_predict(
            dataframe[columns]
        )

        return dataframe.loc[
            prediction == 1
        ].reset_index(drop=True)
    
    def _lof(
        self,
        dataframe: pd.DataFrame,
        columns,
    ) -> pd.DataFrame:

        model = LocalOutlierFactor(
            contamination="auto",
            n_neighbors=20,
            n_jobs=-1,
        )

        prediction = model.fit_predict(
            dataframe[columns]
        )

        return dataframe.loc[
            prediction == 1
        ].reset_index(drop=True)


    def detect(
        self,
        dataframe: pd.DataFrame,
        method: str = "auto",
    ) -> pd.DataFrame:

        cleaned = self.clean(
            dataframe,
            method=method,
        )

        removed_index = dataframe.index.difference(
            cleaned.index
        )

        return dataframe.loc[
            removed_index
        ].copy()


    def statistics(
        self,
    ) -> Dict:

        return {
            "Method": self.report["method"],
            "Rows Before": self.report["rows_before"],
            "Rows After": self.report["rows_after"],
            "Removed Rows": self.report["removed_rows"],
            "Outlier Percentage": self.report["outlier_percentage"],
        }
    
    def summary(
        self,
    ) -> Dict:

        logger.info("=" * 70)
        logger.info("OUTLIER DETECTOR REPORT")
        logger.info("=" * 70)

        for key, value in self.report.items():

            logger.info(f"{key} : {value}")

        logger.info("=" * 70)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/outlier_report.csv",
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
            f"Outlier report saved to {output_path}"
        )


    def get_outlier_indices(
        self,
        dataframe: pd.DataFrame,
        method: str = "auto",
    ) -> list[int]:

        cleaned = self.clean(
            dataframe,
            method=method,
        )

        return dataframe.index.difference(
            cleaned.index
        ).tolist()