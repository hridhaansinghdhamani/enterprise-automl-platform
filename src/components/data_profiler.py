"""
Enterprise Batch Prediction Pipeline
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.common import load_object


class BatchPrediction:
    """
    Batch prediction pipeline for CSV datasets.
    """

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
    ) -> None:

        self.model = load_object(model_path)
        self.preprocessor = load_object(preprocessor_path)

    def predict(
        self,
        input_csv: str,
        output_csv: str,
    ) -> str:
        """
        Predict complete dataset.

        Args:
            input_csv: Input CSV path.
            output_csv: Output CSV path.

        Returns:
            Output CSV path.
        """

        df = pd.read_csv(input_csv)

        transformed = self.preprocessor.transform(df)

        prediction = self.model.predict(transformed)

        result = df.copy()

        result["Prediction"] = prediction

        if hasattr(self.model, "predict_proba"):

            probability = self.model.predict_proba(
                transformed
            )

            if probability.shape[1] >= 2:

                result["Probability"] = probability[:, 1]

        Path(output_csv).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.to_csv(
            output_csv,
            index=False,
        )

        return output_csv