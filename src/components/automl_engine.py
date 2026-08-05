"""
Enterprise AutoML Engine
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from src.components.model_trainer import ModelTrainer
from src.components.problem_detector import ProblemDetector
from src.components.smote_handler import SMOTEHandler
from src.configuration.configuration import ConfigurationManager
from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


class AutoMLEngine:
    """
    Enterprise AutoML Engine

    Responsibilities
    ----------------
    - Detect ML problem type
    - Apply SMOTE for classification
    - Train models
    """

    def __init__(self, config):

        try:

            self.config = config

            self.params = (
                ConfigurationManager()
                .params
            )

            self.problem_detector = (
                ProblemDetector()
            )

            self.smote = SMOTEHandler()

            self.trainer = ModelTrainer(
                config
            )

            logger.info(
                "AutoML Engine initialized successfully."
            )

        except Exception as e:

            logger.exception(
                "Failed to initialize AutoML Engine."
            )

            raise AutoMLException(e)

    def run(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        """
        Execute AutoML workflow.
        """

        try:

            logger.info(
                "Detecting problem type..."
            )

            task = self.problem_detector.detect(
                y_train
            )

            logger.info(
                "Detected task: %s",
                task,
            )

            if task == "classification":

                logger.info(
                    "Applying SMOTE..."
                )

                X_train, y_train = (
                    self.smote.apply(
                        X_train,
                        y_train,
                    )
                )

            logger.info(
                "Starting model training..."
            )

            result = self.trainer.train(
                X_train,
                y_train,
                X_test,
                y_test,
                task,
                cv=self.params["cv"],
                n_trials=self.params["n_trials"],
            )

            logger.info(
                "AutoML workflow completed successfully."
            )

            return result

        except Exception as e:

            logger.exception(
                "AutoML Engine failed."
            )

            raise AutoMLException(e)