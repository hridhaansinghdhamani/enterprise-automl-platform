"""
Enterprise Feature Generator
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures

from src.logger.logger import logger


class FeatureGenerator:
    """
    Enterprise Feature Generator

    Features
    --------
    ✓ Polynomial Features
    ✓ Interaction Features
    ✓ Log Features
    ✓ Square Root Features
    ✓ Ratio Features
    """

    def __init__(self):

        self.report = {
            "generated_features": [],
            "original_features": 0,
            "final_features": 0,
        }

    def generate(

        self,

        dataframe: pd.DataFrame,

        degree: int = 2,

        interaction_only: bool = False,

    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Feature Generation Started")
        logger.info("=" * 70)

        df = dataframe.copy()

        self.report["original_features"] = df.shape[1]

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) < 2:

            return df

        poly = PolynomialFeatures(

            degree=degree,

            include_bias=False,

            interaction_only=interaction_only,

        )

        transformed = poly.fit_transform(

            df[numeric_columns]

        )

        feature_names = poly.get_feature_names_out(

            numeric_columns

        )

        poly_df = pd.DataFrame(

            transformed,

            columns=feature_names,

            index=df.index,

        )

        new_columns = [

            column

            for column in poly_df.columns

            if column not in df.columns

        ]

        df = pd.concat(

            [

                df,

                poly_df[new_columns],

            ],

            axis=1,

        )

        self.report["generated_features"] = new_columns

        self.report["final_features"] = df.shape[1]

        logger.info(

            f"Generated {len(new_columns)} Features"

        )

        return df
    
    """
Enterprise Feature Generator
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures

from src.logger.logger import logger


class FeatureGenerator:
    """
    Enterprise Feature Generator

    Features
    --------
    ✓ Polynomial Features
    ✓ Interaction Features
    ✓ Log Features
    ✓ Square Root Features
    ✓ Ratio Features
    """

    def __init__(self):

        self.report = {
            "generated_features": [],
            "original_features": 0,
            "final_features": 0,
        }

    def generate(

        self,

        dataframe: pd.DataFrame,

        degree: int = 2,

        interaction_only: bool = False,

    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Feature Generation Started")
        logger.info("=" * 70)

        df = dataframe.copy()

        self.report["original_features"] = df.shape[1]

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) < 2:

            return df

        poly = PolynomialFeatures(

            degree=degree,

            include_bias=False,

            interaction_only=interaction_only,

        )

        transformed = poly.fit_transform(

            df[numeric_columns]

        )

        feature_names = poly.get_feature_names_out(

            numeric_columns

        )

        poly_df = pd.DataFrame(

            transformed,

            columns=feature_names,

            index=df.index,

        )

        new_columns = [

            column

            for column in poly_df.columns

            if column not in df.columns

        ]

        df = pd.concat(

            [

                df,

                poly_df[new_columns],

            ],

            axis=1,

        )

        self.report["generated_features"] = new_columns

        self.report["final_features"] = df.shape[1]

        logger.info(

            f"Generated {len(new_columns)} Features"

        )

        return df
    
    def add_log_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:

            if (df[column] >= 0).all():

                new_column = f"{column}_log"

                df[new_column] = np.log1p(
                    df[column]
                )

                self.report[
                    "generated_features"
                ].append(new_column)

        logger.info(
            "Log Features Generated."
        )

        return df


    def add_sqrt_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:

            if (df[column] >= 0).all():

                new_column = f"{column}_sqrt"

                df[new_column] = np.sqrt(
                    df[column]
                )

                self.report[
                    "generated_features"
                ].append(new_column)

        logger.info(
            "Square Root Features Generated."
        )

        return df


    def add_ratio_features(
        self,
        dataframe: pd.DataFrame,
        max_pairs: int = 20,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        pair_count = 0

        for i in range(len(numeric_columns)):

            for j in range(i + 1, len(numeric_columns)):

                if pair_count >= max_pairs:

                    break

                col1 = numeric_columns[i]
                col2 = numeric_columns[j]

                if (df[col2] == 0).any():

                    continue

                new_column = f"{col1}_div_{col2}"

                df[new_column] = (
                    df[col1] / df[col2]
                )

                self.report[
                    "generated_features"
                ].append(new_column)

                pair_count += 1

        logger.info(
            f"Ratio Features Generated : {pair_count}"
        )

        return df
    
    def add_interaction_features(
        self,
        dataframe: pd.DataFrame,
        max_pairs: int = 20,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        pair_count = 0

        for i in range(len(numeric_columns)):

            for j in range(i + 1, len(numeric_columns)):

                if pair_count >= max_pairs:

                    break

                col1 = numeric_columns[i]
                col2 = numeric_columns[j]

                new_column = f"{col1}_x_{col2}"

                df[new_column] = (
                    df[col1] * df[col2]
                )

                self.report[
                    "generated_features"
                ].append(new_column)

                pair_count += 1

        logger.info(
            f"Interaction Features Generated : {pair_count}"
        )

        return df


    def add_date_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        object_columns = df.select_dtypes(
            include=["object"]
        ).columns

        for column in object_columns:

            try:

                converted = pd.to_datetime(
                    df[column],
                    errors="raise",
                )

                df[f"{column}_year"] = converted.dt.year
                df[f"{column}_month"] = converted.dt.month
                df[f"{column}_day"] = converted.dt.day
                df[f"{column}_weekday"] = converted.dt.weekday
                df[f"{column}_quarter"] = converted.dt.quarter

                self.report[
                    "generated_features"
                ].extend(
                    [
                        f"{column}_year",
                        f"{column}_month",
                        f"{column}_day",
                        f"{column}_weekday",
                        f"{column}_quarter",
                    ]
                )

            except Exception:

                continue

        logger.info(
            "Date Features Generated."
        )

        return df


    def add_cyclic_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        month_columns = [

            col

            for col in df.columns

            if col.endswith("_month")

        ]

        for column in month_columns:

            df[f"{column}_sin"] = np.sin(
                2 * np.pi * df[column] / 12
            )

            df[f"{column}_cos"] = np.cos(
                2 * np.pi * df[column] / 12
            )

            self.report[
                "generated_features"
            ].extend(
                [
                    f"{column}_sin",
                    f"{column}_cos",
                ]
            )

        logger.info(
            "Cyclic Features Generated."
        )

        return df
    
    def summary(
        self,
    ) -> Dict:

        logger.info("=" * 70)
        logger.info("FEATURE GENERATOR REPORT")
        logger.info("=" * 70)

        logger.info(
            f"Original Features : {self.report['original_features']}"
        )

        logger.info(
            f"Final Features : {self.report['final_features']}"
        )

        logger.info(
            f"Generated Features : {len(self.report['generated_features'])}"
        )

        logger.info("=" * 70)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/feature_generator_report.csv",
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
            f"Feature Generator Report saved to {output_path}"
        )


    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        df = self.generate(df)

        df = self.add_log_features(df)

        df = self.add_sqrt_features(df)

        df = self.add_ratio_features(df)

        df = self.add_interaction_features(df)

        df = self.add_date_features(df)

        df = self.add_cyclic_features(df)

        self.report["final_features"] = df.shape[1]

        logger.info(
            f"Total Features After Generation : {df.shape[1]}"
        )

        return df


    def generated_feature_names(
        self,
    ) -> list[str]:

        return self.report["generated_features"]


    def reset(
        self,
    ) -> None:

        self.report = {
            "generated_features": [],
            "original_features": 0,
            "final_features": 0,
        }

        logger.info(
            "Feature Generator Reset Successfully."
        )