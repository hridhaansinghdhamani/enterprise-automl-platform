"""
Enterprise Feature Importance Engine
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

from src.logger.logger import logger


class FeatureImportance:
    """
    Enterprise Feature Importance Engine

    Supports
    --------
    ✓ Tree Feature Importance
    ✓ Permutation Importance
    ✓ Mutual Information
    ✓ SHAP (future integration)
    """

    def __init__(self):

        self.report = {
            "method": None,
            "top_features": [],
        }

    def calculate(

        self,

        model,

        X: pd.DataFrame,

        y,

        task: str,

        method: str = "auto",

    ) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Calculating Feature Importance")
        logger.info("=" * 70)

        if method == "auto":

            if hasattr(
                model,
                "feature_importances_",
            ):

                method = "tree"

            else:

                method = "permutation"

        self.report["method"] = method

        if method == "tree":

            importance = pd.DataFrame(

                {

                    "Feature": X.columns,

                    "Importance": model.feature_importances_,

                }

            )

        elif method == "permutation":

            result = permutation_importance(

                model,

                X,

                y,

                random_state=42,

                n_repeats=10,

                n_jobs=-1,

            )

            importance = pd.DataFrame(

                {

                    "Feature": X.columns,

                    "Importance": result.importances_mean,

                }

            )

        else:

            scores = (

                mutual_info_classif(

                    X,

                    y,

                    random_state=42,

                )

                if task == "classification"

                else mutual_info_regression(

                    X,

                    y,

                    random_state=42,

                )

            )

            importance = pd.DataFrame(

                {

                    "Feature": X.columns,

                    "Importance": scores,

                }

            )

        importance = importance.sort_values(

            by="Importance",

            ascending=False,

        ).reset_index(drop=True)

        self.report["top_features"] = (

            importance.head(20)["Feature"]

            .tolist()

        )

        logger.info(

            "Feature Importance Completed"

        )

        return importance
    
    def top_features(
        self,
        importance: pd.DataFrame,
        top_n: int = 20,
    ) -> pd.DataFrame:

        return (
            importance
            .head(top_n)
            .reset_index(drop=True)
        )


    def cumulative_importance(
        self,
        importance: pd.DataFrame,
    ) -> pd.DataFrame:

        df = importance.copy()

        total = df["Importance"].sum()

        if total != 0:

            df["Normalized Importance"] = (
                df["Importance"] / total
            )

            df["Cumulative Importance"] = (
                df["Normalized Importance"]
                .cumsum()
            )

        else:

            df["Normalized Importance"] = 0.0

            df["Cumulative Importance"] = 0.0

        return df


    def export_importance(
        self,
        importance: pd.DataFrame,
        output_path: str = "reports/feature_importance.csv",
    ) -> None:

        import os

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True,
        )

        importance.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            f"Feature importance exported to {output_path}"
        )


    def get_selected_features(
        self,
        importance: pd.DataFrame,
        threshold: float = 0.01,
    ) -> list[str]:

        return (

            importance[
                importance["Importance"] >= threshold
            ]["Feature"]

            .tolist()

        )
    def summary(
        self,
    ) -> Dict:

        logger.info("=" * 70)
        logger.info("FEATURE IMPORTANCE REPORT")
        logger.info("=" * 70)

        for key, value in self.report.items():

            logger.info(f"{key} : {value}")

        logger.info("=" * 70)

        return self.report


    def export_report(
        self,
        output_path: str = "reports/feature_importance_report.csv",
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
            f"Feature importance report saved to {output_path}"
        )


    def fit(
        self,
        model,
        X: pd.DataFrame,
        y,
        task: str,
        method: str = "auto",
    ) -> None:

        self.calculate(
            model=model,
            X=X,
            y=y,
            task=task,
            method=method,
        )


    def fit_transform(
        self,
        model,
        X: pd.DataFrame,
        y,
        task: str,
        method: str = "auto",
    ) -> pd.DataFrame:

        return self.calculate(
            model=model,
            X=X,
            y=y,
            task=task,
            method=method,
        )


    def reset(
        self,
    ) -> None:

        self.report = {
            "method": None,
            "top_features": [],
        }

        logger.info(
            "Feature Importance Reset Successfully."
        )