"""
Enterprise Problem Detector

Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import pandas as pd

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


class ProblemDetector:
    """
    Detect whether the target variable represents a
    classification or regression problem.
    """

    def detect(
        self,
        target: pd.Series,
    ) -> str:
        """
        Detect machine learning problem type.

        Parameters
        ----------
        target : pd.Series
            Target column.

        Returns
        -------
        str
            "classification" or "regression"
        """

        try:

            if target.empty:
                raise ValueError(
                    "Target column is empty."
                )

            unique = target.nunique()

            if (
                target.dtype == "object"
                or str(target.dtype) == "category"
                or str(target.dtype) == "bool"
            ):

                logger.info(
                    "Problem detected: Classification"
                )

                return "classification"

            if unique <= 10:

                logger.info(
                    "Problem detected: Classification"
                )

                return "classification"

            logger.info(
                "Problem detected: Regression"
            )

            return "regression"

        except Exception as e:

            logger.exception(
                "Failed to detect problem type."
            )

            raise AutoMLException(e)