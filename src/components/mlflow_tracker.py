"""
Enterprise MLflow Tracker

Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracking utility.
    """

    def __init__(
        self,
        experiment_name: str = "Enterprise AutoML Platform",
    ):

        self.experiment_name = experiment_name

    def start(self) -> None:
        """
        Start an MLflow run.
        """

        try:

            if mlflow.active_run() is not None:
                mlflow.end_run()

            mlflow.set_experiment(
                self.experiment_name,
            )

            mlflow.start_run()

            logger.info(
                "MLflow run started."
            )

        except Exception as e:

            logger.exception(
                "Failed to start MLflow run."
            )

            raise AutoMLException(e)

    def log_params(
        self,
        params: dict,
    ) -> None:
        """
        Log model parameters.
        """

        try:

            for key, value in params.items():

                if value is not None:

                    mlflow.log_param(
                        key,
                        value,
                    )

        except Exception as e:

            logger.exception(
                "Failed to log MLflow parameters."
            )

            raise AutoMLException(e)

    def log_metrics(
        self,
        metrics: dict,
    ) -> None:
        """
        Log evaluation metrics.
        """

        try:

            for key, value in metrics.items():

                if value is not None:

                    mlflow.log_metric(
                        key,
                        float(value),
                    )

        except Exception as e:

            logger.exception(
                "Failed to log MLflow metrics."
            )

            raise AutoMLException(e)

    def log_model(
        self,
        model,
    ) -> None:
        """
        Log trained model.
        """

        try:

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )

            logger.info(
                "Model logged to MLflow."
            )

        except Exception as e:

            logger.exception(
                "Failed to log model."
            )

            raise AutoMLException(e)

    def end(self) -> None:
        """
        End current MLflow run.
        """

        try:

            if mlflow.active_run() is not None:

                mlflow.end_run()

                logger.info(
                    "MLflow run ended."
                )

        except Exception as e:

            logger.exception(
                "Failed to end MLflow run."
            )

            raise AutoMLException(e)