"""
Enterprise Batch Prediction Pipeline

Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger
from src.utils.common import load_object

logger = get_logger(__name__)


class BatchPrediction:
    """
    Batch prediction pipeline for CSV datasets.
    """

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
    ) -> None:

        try:

            model_file = Path(model_path)

            if not model_file.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}"
                )

            preprocessor_file = Path(preprocessor_path)

            if not preprocessor_file.exists():
                raise FileNotFoundError(
                    f"Preprocessor file not found: {preprocessor_path}"
                )

            self.model = load_object(str(model_file))
            self.preprocessor = load_object(str(preprocessor_file))

            logger.info(
                "Batch Prediction initialized successfully."
            )

        except Exception as e:

            logger.exception(
                "Failed to initialize BatchPrediction."
            )

            raise AutoMLException(e)

    def predict(
        self,
        input_csv: str,
        output_csv: str,
    ) -> str:
        """
        Predict complete dataset.

        Parameters
        ----------
        input_csv : str
            Input CSV file path.

        output_csv : str
            Output CSV file path.

        Returns
        -------
        str
            Output CSV path.
        """

        try:

            input_file = Path(input_csv)

            if not input_file.exists():
                raise FileNotFoundError(
                    f"Input CSV not found: {input_csv}"
                )

            logger.info(
                "Reading input dataset..."
            )

            df = pd.read_csv(input_file)

            if df.empty:
                raise ValueError(
                    "Input CSV is empty."
                )

            logger.info(
                "Preprocessing dataset..."
            )

            transformed = self.preprocessor.transform(df)

            logger.info(
                "Generating predictions..."
            )

            prediction = self.model.predict(transformed)

            result = df.copy()

            result["Prediction"] = prediction

            if hasattr(self.model, "predict_proba"):

                probability = self.model.predict_proba(
                    transformed
                )

                if (
                    probability.ndim == 2
                    and probability.shape[1] >= 2
                ):

                    result["Probability"] = probability[:, 1]

            output_path = Path(output_csv)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            result.to_csv(
                output_path,
                index=False,
            )

            logger.info(
                "Batch prediction completed successfully."
            )

            logger.info(
                "Predictions saved to %s",
                output_path,
            )

            return str(output_path)

        except Exception as e:

            logger.exception(
                "Batch prediction failed."
            )

            raise AutoMLException(e)