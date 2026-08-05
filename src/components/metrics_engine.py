"""
Enterprise Metrics Engine

Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


class MetricsEngine:
    """
    Compute evaluation metrics for
    classification and regression models.
    """

    def classification(
        self,
        y_true,
        y_pred,
        y_prob=None,
    ) -> dict:
        """
        Calculate classification metrics.
        """

        try:

            metrics = {
                "Accuracy": accuracy_score(
                    y_true,
                    y_pred,
                ),
                "Precision": precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
            }

            try:

                if (
                    y_prob is not None
                    and len(np.unique(y_true)) == 2
                ):

                    metrics["ROC_AUC"] = roc_auc_score(
                        y_true,
                        y_prob,
                    )

                else:

                    metrics["ROC_AUC"] = None

            except Exception:

                metrics["ROC_AUC"] = None

            logger.info(
                "Classification metrics calculated successfully."
            )

            return metrics

        except Exception as e:

            logger.exception(
                "Failed to calculate classification metrics."
            )

            raise AutoMLException(e)

    def regression(
        self,
        y_true,
        y_pred,
    ) -> dict:
        """
        Calculate regression metrics.
        """

        try:

            mse = mean_squared_error(
                y_true,
                y_pred,
            )

            metrics = {
                "R2": r2_score(
                    y_true,
                    y_pred,
                ),
                "MAE": mean_absolute_error(
                    y_true,
                    y_pred,
                ),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
            }

            logger.info(
                "Regression metrics calculated successfully."
            )

            return metrics

        except Exception as e:

            logger.exception(
                "Failed to calculate regression metrics."
            )

            raise AutoMLException(e)