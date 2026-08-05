"""
Enterprise Feature Selector
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import os
from typing import Dict

import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:

    def __init__(self):
        self.report = {
            "original_features": 0,
            "selected_features": 0,
            "removed_features": [],
            "selection_method": None,
        }

    def select(self, X: pd.DataFrame, y, task: str,
               method: str = "mutual_info",
               k: int | str = "all") -> pd.DataFrame:
        try:
            self.report["original_features"] = X.shape[1]

            if method == "variance":
                selector = VarianceThreshold()
                transformed = selector.fit_transform(X)
                columns = X.columns[selector.get_support()]
                self.report["selection_method"] = "VarianceThreshold"
            else:
                score_function = (
                    mutual_info_classif
                    if task == "classification"
                    else mutual_info_regression
                )
                selector = SelectKBest(score_func=score_function, k=k)
                transformed = selector.fit_transform(X, y)
                columns = X.columns[selector.get_support()]
                self.report["selection_method"] = "Mutual Information"

            selected = pd.DataFrame(
                transformed,
                columns=columns,
                index=X.index,
            )

            self.report["selected_features"] = selected.shape[1]
            self.report["removed_features"] = list(
                set(X.columns) - set(selected.columns)
            )

            return selected

        except Exception as e:
            logger.exception("Feature selection failed.")
            raise AutoMLException(e)

    def feature_scores(self, X: pd.DataFrame, y, task: str) -> pd.DataFrame:
        try:
            score_function = (
                mutual_info_classif
                if task == "classification"
                else mutual_info_regression
            )

            selector = SelectKBest(score_func=score_function, k="all")
            selector.fit(X, y)

            return (
                pd.DataFrame(
                    {
                        "Feature": X.columns,
                        "Score": selector.scores_,
                    }
                )
                .sort_values("Score", ascending=False)
                .reset_index(drop=True)
            )

        except Exception as e:
            logger.exception("Failed to calculate feature scores.")
            raise AutoMLException(e)

    def variance_report(self, X: pd.DataFrame) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "Feature": X.columns,
                    "Variance": [X[c].var() for c in X.columns],
                }
            )
            .sort_values("Variance", ascending=False)
            .reset_index(drop=True)
        )

    def feature_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Feature": X.columns,
                "Data Type": [str(X[c].dtype) for c in X.columns],
                "Missing": [int(X[c].isna().sum()) for c in X.columns],
                "Unique": [int(X[c].nunique()) for c in X.columns],
            }
        )

    def export_scores(
        self,
        X: pd.DataFrame,
        y,
        task: str,
        output_path: str = "reports/feature_scores.csv",
    ) -> None:

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            scores = self.feature_scores(X, y, task)
            scores.to_csv(output_path, index=False)

        except Exception as e:
            logger.exception("Failed to export feature scores.")
            raise AutoMLException(e)

    def summary(self) -> Dict:
        return self.report

    def export_report(
        self,
        output_path: str = "reports/feature_selector_report.csv",
    ) -> None:

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            pd.DataFrame(
                [
                    {"Metric": k, "Value": str(v)}
                    for k, v in self.report.items()
                ]
            ).to_csv(output_path, index=False)

        except Exception as e:
            logger.exception("Failed to export report.")
            raise AutoMLException(e)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y,
        task: str,
        method: str = "mutual_info",
        k: int | str = "all",
    ) -> pd.DataFrame:
        return self.select(X, y, task, method, k)

    def get_selected_features(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        return dataframe.columns.tolist()