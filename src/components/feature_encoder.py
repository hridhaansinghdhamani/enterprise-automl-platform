"""
Enterprise Feature Encoder
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import joblib
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)

from src.logger.logger import logger


class FeatureEncoder:
    """
    Enterprise Feature Encoding Engine

    Supports
    --------
    ✓ Label Encoding
    ✓ One Hot Encoding
    ✓ Ordinal Encoding
    ✓ Frequency Encoding
    ✓ Automatic Encoding
    """

    def __init__(self):

        self.encoders = {}

        self.report = {
            "method": None,
            "encoded_columns": [],
        }

    def fit_transform(

        self,

        dataframe: pd.DataFrame,

        method: str = "auto",

    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Feature Encoding Started")
        logger.info("=" * 70)

        df = dataframe.copy()

        categorical_columns = df.select_dtypes(

            include=["object", "category"]

        ).columns.tolist()

        if len(categorical_columns) == 0:

            return df

        if method == "auto":

            method = "label"

        self.report["method"] = method

        if method == "label":

            for column in categorical_columns:

                encoder = LabelEncoder()

                df[column] = encoder.fit_transform(

                    df[column].astype(str)

                )

                self.encoders[column] = encoder

        elif method == "ordinal":

            encoder = OrdinalEncoder(

                handle_unknown="use_encoded_value",

                unknown_value=-1,

            )

            df[categorical_columns] = encoder.fit_transform(

                df[categorical_columns]

            )

            self.encoders["ordinal"] = encoder

        elif method == "onehot":

            encoder = OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=False,

            )

            transformed = encoder.fit_transform(

                df[categorical_columns]

            )

            feature_names = encoder.get_feature_names_out(

                categorical_columns

            )

            encoded_df = pd.DataFrame(

                transformed,

                columns=feature_names,

                index=df.index,

            )

            df = pd.concat(

                [

                    df.drop(columns=categorical_columns),

                    encoded_df,

                ],

                axis=1,

            )

            self.encoders["onehot"] = encoder

        self.report["encoded_columns"] = categorical_columns

        logger.info(

            f"Encoded {len(categorical_columns)} Columns"

        )

        return df
    
    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        if self.report["method"] == "label":

            for column, encoder in self.encoders.items():

                if column not in df.columns:

                    continue

                values = df[column].astype(str)

                mapping = {
                    cls: idx
                    for idx, cls in enumerate(
                        encoder.classes_
                    )
                }

                df[column] = values.map(mapping).fillna(-1).astype(int)

        elif self.report["method"] == "ordinal":

            encoder = self.encoders["ordinal"]

            categorical = df.select_dtypes(
                include=["object", "category"]
            ).columns

            df[categorical] = encoder.transform(
                df[categorical]
            )

        elif self.report["method"] == "onehot":

            encoder = self.encoders["onehot"]

            categorical = df.select_dtypes(
                include=["object", "category"]
            ).columns

            transformed = encoder.transform(
                df[categorical]
            )

            encoded = pd.DataFrame(

                transformed,

                columns=encoder.get_feature_names_out(
                    categorical
                ),

                index=df.index,

            )

            df = pd.concat(

                [

                    df.drop(columns=categorical),

                    encoded,

                ],

                axis=1,

            )

        return df

    def inverse_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        if self.report["method"] != "label":

            return df

        for column, encoder in self.encoders.items():

            if column not in df.columns:

                continue

            mapping = {
                idx: cls
                for idx, cls in enumerate(
                    encoder.classes_
                )
            }

            df[column] = (
                df[column]
                .map(mapping)
                .fillna("Unknown")
            )

        return df


    def save(
        self,
        file_path: str,
    ) -> None:

        joblib.dump(
            self.encoders,
            file_path,
        )

        logger.info(
            f"Encoder saved to {file_path}"
        )


    def load(
        self,
        file_path: str,
    ) -> None:

        self.encoders = joblib.load(
            file_path
        )

        logger.info(
            f"Encoder loaded from {file_path}"
        )

    def frequency_encode(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns

        for column in categorical_columns:

            frequency = (
                df[column]
                .value_counts(normalize=True)
                .to_dict()
            )

            df[column] = df[column].map(
                frequency
            )

            self.encoders[
                f"{column}_frequency"
            ] = frequency

        self.report["method"] = "frequency"

        self.report[
            "encoded_columns"
        ] = categorical_columns.tolist()

        logger.info(
            f"Frequency Encoded {len(categorical_columns)} Columns"
        )

        return df


    def target_encode(
        self,
        dataframe: pd.DataFrame,
        target,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns

        for column in categorical_columns:

            mapping = (
                pd.DataFrame(
                    {
                        column: df[column],
                        "__target__": target,
                    }
                )
                .groupby(column)["__target__"]
                .mean()
                .to_dict()
            )

            df[column] = df[column].map(
                mapping
            )

            self.encoders[
                f"{column}_target"
            ] = mapping

        self.report["method"] = "target"

        self.report[
            "encoded_columns"
        ] = categorical_columns.tolist()

        logger.info(
            f"Target Encoded {len(categorical_columns)} Columns"
        )

        return df


    def available_methods(
        self,
    ) -> list[str]:

        return [

            "label",

            "ordinal",

            "onehot",

            "frequency",

            "target",

            "auto",

        ]


    def reset(
        self,
    ) -> None:

        self.encoders = {}

        self.report = {

            "method": None,

            "encoded_columns": [],

        }

        logger.info(
            "Feature Encoder Reset Successfully."
        )

    def summary(
        self,
    ) -> Dict:

        logger.info("=" * 70)
        logger.info("FEATURE ENCODER REPORT")
        logger.info("=" * 70)

        for key, value in self.report.items():

            logger.info(f"{key} : {value}")

        logger.info("=" * 70)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/feature_encoder_report.csv",
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
            f"Feature encoder report saved to {output_path}"
        )


    def get_feature_names(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:

        return dataframe.columns.tolist()


    def fit(
        self,
        dataframe: pd.DataFrame,
        method: str = "auto",
    ) -> None:

        self.fit_transform(
            dataframe=dataframe,
            method=method,
        )


    def fit_transform_with_target(
        self,
        dataframe: pd.DataFrame,
        target,
        method: str = "target",
    ) -> pd.DataFrame:

        if method == "target":

            return self.target_encode(
                dataframe,
                target,
            )

        return self.fit_transform(
            dataframe,
            method=method,
        )

