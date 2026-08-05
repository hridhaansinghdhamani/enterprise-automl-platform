"""
Enterprise Prediction Service
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger
from src.utils.common import load_object

logger = get_logger(__name__)


class PredictionService:
    """
    Enterprise Prediction Service.
    Responsible for loading the trained model,
    preprocessing input data and generating predictions.
    """

    def __init__(self):

        try:

            # -------------------------
            # Load Model
            # -------------------------
            model_path = Path("saved_models/latest.pkl")

            if not model_path.exists():
                model_path = Path(
                    "artifacts/model_trainer/model.pkl"
                )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}"
                )

            self.model = load_object(str(model_path))

            # -------------------------
            # Load Preprocessor
            # -------------------------
            preprocessor_path = Path(
                "artifacts/data_transformation/preprocessor.pkl"
            )

            if not preprocessor_path.exists():
                raise FileNotFoundError(
                    f"Preprocessor not found: {preprocessor_path}"
                )

            self.preprocessor = load_object(
                str(preprocessor_path)
            )

            logger.info(
                "Prediction service initialized successfully."
            )

        except Exception as e:

            logger.exception(
                "Failed to initialize PredictionService."
            )

            raise AutoMLException(e)

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions for input dataframe.
        """

        try:

            if dataframe.empty:
                raise ValueError(
                    "Input dataframe is empty."
                )

            logger.info(
                "Preprocessing input data..."
            )

            transformed = self.preprocessor.transform(
                dataframe
            )

            logger.info(
                "Generating predictions..."
            )

            prediction = self.model.predict(
                transformed
            )

            result = dataframe.copy()

            result["Prediction"] = prediction

            # -------------------------
            # Classification Probability
            # -------------------------
            if hasattr(
                self.model,
                "predict_proba",
            ):

                probability = self.model.predict_proba(
                    transformed
                )

                if probability.ndim == 2 and probability.shape[1] >= 2:

                    result["Probability"] = probability[:, 1]

            logger.info(
                "Prediction completed successfully."
            )

            return result

        except Exception as e:

            logger.exception(
                "Prediction failed."
            )

            raise AutoMLException(e)