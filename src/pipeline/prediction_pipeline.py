"""
Enterprise Prediction Pipeline
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import time

import pandas as pd

from src.components.model_monitor import ModelMonitor
from src.components.prediction_service import PredictionService
from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


class PredictionPipeline:
    """
    Enterprise Prediction Pipeline
    """

    def __init__(self):

        self.service = PredictionService()
        self.monitor = ModelMonitor()

    def predict(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Perform prediction and log inference metrics.
        """

        try:

            if dataframe.empty:
                raise ValueError(
                    "Input dataframe is empty."
                )

            logger.info("Starting prediction...")

            start = time.perf_counter()

            result = self.service.predict(
                dataframe
            )

            inference_time = (
                time.perf_counter() - start
            ) * 1000

            if isinstance(result, pd.DataFrame):

                if (
                    "Prediction" in result.columns
                    and not result.empty
                ):

                    prediction = (
                        result["Prediction"]
                        .mode()
                        .iloc[0]
                    )

                    probability = 0.0

                    if (
                        "Probability" in result.columns
                        and not result["Probability"].empty
                    ):

                        probability = float(
                            result["Probability"].mean()
                        )

                    self.monitor.log_prediction(
                        prediction=prediction,
                        probability=probability,
                        inference_time=inference_time,
                    )

            elif (
                isinstance(result, list)
                and len(result) > 0
            ):

                self.monitor.log_prediction(
                    prediction=result[0],
                    probability=0.0,
                    inference_time=inference_time,
                )

            logger.info(
                "Prediction completed successfully in %.2f ms.",
                inference_time,
            )

            return result

        except Exception as e:

            logger.exception(
                "Prediction pipeline failed."
            )

            raise AutoMLException(e)