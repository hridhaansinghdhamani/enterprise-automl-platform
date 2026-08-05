"""
Enterprise Feature Reducer
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import joblib
import pandas as pd

from sklearn.decomposition import (
    PCA,
    IncrementalPCA,
    TruncatedSVD,
)

from src.logger.logger import logger


class FeatureReducer:
    """
    Enterprise Feature Reduction Engine

    Supports
    --------
    ✓ PCA
    ✓ Incremental PCA
    ✓ Truncated SVD
    ✓ Auto Component Selection
    """

    def __init__(self):

        self.reducer = None

        self.report = {

            "method": None,

            "original_features": 0,

            "reduced_features": 0,

            "explained_variance": 0,

        }

    def fit_transform(

        self,

        dataframe: pd.DataFrame,

        method: str = "pca",

        n_components=0.95,

    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Feature Reduction Started")
        logger.info("=" * 70)

        df = dataframe.copy()

        self.report["original_features"] = df.shape[1]

        reducer_map = {

            "pca": PCA(
                n_components=n_components,
                random_state=42,
            ),

            "incremental": IncrementalPCA(
                n_components=min(
                    100,
                    df.shape[1],
                )
            ),

            "svd": TruncatedSVD(
                n_components=min(
                    100,
                    df.shape[1] - 1,
                ),
                random_state=42,
            ),

        }

        if method not in reducer_map:

            method = "pca"

        self.reducer = reducer_map[method]

        transformed = self.reducer.fit_transform(df)

        columns = [

            f"PC_{i+1}"

            for i in range(

                transformed.shape[1]

            )

        ]

        reduced_df = pd.DataFrame(

            transformed,

            columns=columns,

            index=df.index,

        )

        self.report["method"] = method

        self.report["reduced_features"] = reduced_df.shape[1]

        if hasattr(

            self.reducer,

            "explained_variance_ratio_",

        ):

            self.report["explained_variance"] = round(

                self.reducer.explained_variance_ratio_.sum(),

                4,

            )

        logger.info(

            f"Reduced {df.shape[1]} → {reduced_df.shape[1]} Features"

        )

        return reduced_df
    
    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.reducer is None:

            raise ValueError(
                "Reducer has not been fitted."
            )

        transformed = self.reducer.transform(
            dataframe
        )

        columns = [

            f"PC_{i+1}"

            for i in range(
                transformed.shape[1]
            )

        ]

        return pd.DataFrame(

            transformed,

            columns=columns,

            index=dataframe.index,

        )


    def inverse_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.reducer is None:

            raise ValueError(
                "Reducer has not been fitted."
            )

        if not hasattr(
            self.reducer,
            "inverse_transform",
        ):

            return dataframe

        reconstructed = self.reducer.inverse_transform(
            dataframe
        )

        columns = [

            f"Feature_{i+1}"

            for i in range(
                reconstructed.shape[1]
            )

        ]

        return pd.DataFrame(

            reconstructed,

            columns=columns,

            index=dataframe.index,

        )


    def save(
        self,
        file_path: str,
    ) -> None:

        joblib.dump(
            self.reducer,
            file_path,
        )

        logger.info(
            f"Reducer saved to {file_path}"
        )


    def load(
        self,
        file_path: str,
    ) -> None:

        self.reducer = joblib.load(
            file_path
        )

        logger.info(
            f"Reducer loaded from {file_path}"
        )

    def explained_variance_report(
        self,
    ) -> pd.DataFrame:

        if self.reducer is None:

            return pd.DataFrame()

        if not hasattr(
            self.reducer,
            "explained_variance_ratio_",
        ):

            return pd.DataFrame()

        variance = self.reducer.explained_variance_ratio_

        cumulative = variance.cumsum()

        return pd.DataFrame(

            {

                "Component": [

                    f"PC_{i+1}"

                    for i in range(

                        len(variance)

                    )

                ],

                "Explained Variance": variance,

                "Cumulative Variance": cumulative,

            }

        )


    def scree_plot_data(
        self,
    ) -> pd.DataFrame:

        return self.explained_variance_report()


    def available_methods(
        self,
    ) -> list[str]:

        return [

            "pca",

            "incremental",

            "svd",

        ]


    def fit(
        self,
        dataframe: pd.DataFrame,
        method: str = "pca",
        n_components=0.95,
    ) -> None:

        self.fit_transform(

            dataframe,

            method,

            n_components,

        )


    def reset(
        self,
    ) -> None:

        self.reducer = None

        self.report = {

            "method": None,

            "original_features": 0,

            "reduced_features": 0,

            "explained_variance": 0,

        }

        logger.info(
            "Feature Reducer Reset Successfully."
        )    

    def summary(
        self,
    ) -> Dict:

        logger.info("=" * 70)
        logger.info("FEATURE REDUCER REPORT")
        logger.info("=" * 70)

        for key, value in self.report.items():

            logger.info(f"{key} : {value}")

        logger.info("=" * 70)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/feature_reducer_report.csv",
    ) -> None:

        import os

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True,
        )

        report = pd.DataFrame(
            [
                {
                    "Metric": key,
                    "Value": str(value),
                }
                for key, value in self.report.items()
            ]
        )

        report.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            f"Feature reducer report saved to {output_path}"
        )


    def get_components(
        self,
    ) -> pd.DataFrame:

        if self.reducer is None:

            return pd.DataFrame()

        if not hasattr(
            self.reducer,
            "components_",
        ):

            return pd.DataFrame()

        columns = [

            f"Feature_{i+1}"

            for i in range(
                self.reducer.components_.shape[1]
            )

        ]

        index = [

            f"PC_{i+1}"

            for i in range(
                self.reducer.components_.shape[0]
            )

        ]

        return pd.DataFrame(
            self.reducer.components_,
            columns=columns,
            index=index,
        )


    def get_feature_importance(
        self,
    ) -> pd.DataFrame:

        components = self.get_components()

        if components.empty:

            return pd.DataFrame()

        importance = (
            components.abs()
            .sum(axis=0)
            .sort_values(
                ascending=False
            )
        )

        return pd.DataFrame(

            {

                "Feature": importance.index,

                "Importance": importance.values,

            }

        )


    def fit_transform_auto(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if dataframe.shape[1] <= 30:

            logger.info(
                "Feature count is low. Skipping reduction."
            )

            return dataframe.copy()

        return self.fit_transform(
            dataframe=dataframe,
            method="pca",
            n_components=0.95,
        )    