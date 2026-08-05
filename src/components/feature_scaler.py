"""
Enterprise Feature Scaler
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import joblib
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
)

from src.logger.logger import logger


class FeatureScaler:
    """
    Enterprise Feature Scaling Engine

    Supports
    --------
    ✓ StandardScaler
    ✓ MinMaxScaler
    ✓ RobustScaler
    ✓ MaxAbsScaler
    ✓ Normalizer
    ✓ QuantileTransformer
    ✓ PowerTransformer
    """

    def __init__(self):

        self.scaler = None

        self.report = {

            "method": None,

            "scaled_columns": [],

        }

    def fit_transform(

        self,

        dataframe: pd.DataFrame,

        method: str = "standard",

    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Feature Scaling Started")
        logger.info("=" * 70)

        df = dataframe.copy()

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_columns) == 0:

            return df

        scaler_map = {

            "standard": StandardScaler(),

            "minmax": MinMaxScaler(),

            "robust": RobustScaler(),

            "maxabs": MaxAbsScaler(),

            "normalize": Normalizer(),

            "quantile": QuantileTransformer(
                random_state=42,
            ),

            "power": PowerTransformer(),

        }

        if method not in scaler_map:

            method = "standard"

        self.scaler = scaler_map[method]

        df[numeric_columns] = self.scaler.fit_transform(

            df[numeric_columns]

        )

        self.report["method"] = method

        self.report["scaled_columns"] = numeric_columns

        logger.info(

            f"Scaled {len(numeric_columns)} Columns using {method}"

        )

        return df
    
    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.scaler is None:

            raise ValueError(
                "Scaler has not been fitted."
            )

        df = dataframe.copy()

        numeric_columns = self.report[
            "scaled_columns"
        ]

        if len(numeric_columns) == 0:

            return df

        df[numeric_columns] = self.scaler.transform(
            df[numeric_columns]
        )

        return df


    def inverse_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.scaler is None:

            raise ValueError(
                "Scaler has not been fitted."
            )

        df = dataframe.copy()

        numeric_columns = self.report[
            "scaled_columns"
        ]

        if hasattr(
            self.scaler,
            "inverse_transform",
        ):

            df[numeric_columns] = self.scaler.inverse_transform(
                df[numeric_columns]
            )

        return df


    def save(
        self,
        file_path: str,
    ) -> None:

        joblib.dump(
            self.scaler,
            file_path,
        )

        logger.info(
            f"Scaler saved to {file_path}"
        )


    def load(
        self,
        file_path: str,
    ) -> None:

        self.scaler = joblib.load(
            file_path,
        )

        logger.info(
            f"Scaler loaded from {file_path}"
        )

    def available_methods(
        self,
    ) -> list[str]:

        return [

            "standard",

            "minmax",

            "robust",

            "maxabs",

            "normalize",

            "quantile",

            "power",

        ]


    def statistics(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        numeric = dataframe.select_dtypes(
            include="number"
        )

        if numeric.empty:

            return pd.DataFrame()

        rows = []

        for column in numeric.columns:

            rows.append(

                {

                    "Feature": column,

                    "Mean": round(
                        numeric[column].mean(),
                        4,
                    ),

                    "Std": round(
                        numeric[column].std(),
                        4,
                    ),

                    "Min": round(
                        numeric[column].min(),
                        4,
                    ),

                    "Max": round(
                        numeric[column].max(),
                        4,
                    ),

                }

            )

        return pd.DataFrame(rows)


    def fit(
        self,
        dataframe: pd.DataFrame,
        method: str = "standard",
    ) -> None:

        self.fit_transform(
            dataframe,
            method=method,
        )


    def reset(
        self,
    ) -> None:

        self.scaler = None

        self.report = {

            "method": None,

            "scaled_columns": [],

        }

        logger.info(
            "Feature Scaler Reset Successfully."
        )    

    def summary(
        self,
    ) -> Dict:

        logger.info("=" * 70)
        logger.info("FEATURE SCALER REPORT")
        logger.info("=" * 70)

        for key, value in self.report.items():

            logger.info(f"{key} : {value}")

        logger.info("=" * 70)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/feature_scaler_report.csv",
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
            f"Feature scaler report saved to {output_path}"
        )


    def get_scaler(self):

        return self.scaler


    def fit_transform_with_columns(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
        method: str = "standard",
    ) -> pd.DataFrame:

        if not columns:

            return dataframe

        df = dataframe.copy()

        scaler_map = {

            "standard": StandardScaler(),

            "minmax": MinMaxScaler(),

            "robust": RobustScaler(),

            "maxabs": MaxAbsScaler(),

            "normalize": Normalizer(),

            "quantile": QuantileTransformer(
                random_state=42,
            ),

            "power": PowerTransformer(),

        }

        self.scaler = scaler_map.get(
            method,
            StandardScaler(),
        )

        df[columns] = self.scaler.fit_transform(
            df[columns]
        )

        self.report["method"] = method
        self.report["scaled_columns"] = columns

        logger.info(
            f"Scaled {len(columns)} Selected Columns."
        )

        return df    