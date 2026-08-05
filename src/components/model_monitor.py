"""
Enterprise Model Monitoring

Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


class ModelMonitor:
    """
    Monitor prediction statistics and inference logs.
    """

    def __init__(
        self,
        log_dir: str = "logs/monitoring",
    ) -> None:

        try:

            self.log_dir = Path(log_dir)

            self.log_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.log_file = (
                self.log_dir
                / "prediction_logs.csv"
            )

            if not self.log_file.exists():

                pd.DataFrame(
                    columns=[
                        "Timestamp",
                        "Prediction",
                        "Probability",
                        "Inference Time (ms)",
                    ]
                ).to_csv(
                    self.log_file,
                    index=False,
                )

            logger.info(
                "Model monitoring initialized."
            )

        except Exception as e:

            logger.exception(
                "Failed to initialize ModelMonitor."
            )

            raise AutoMLException(e)

    def log_prediction(
        self,
        prediction,
        probability,
        inference_time,
    ) -> None:
        """
        Log a prediction record.
        """

        try:

            record = pd.DataFrame(
                [
                    {
                        "Timestamp": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Prediction": prediction,
                        "Probability": probability,
                        "Inference Time (ms)": round(
                            inference_time,
                            2,
                        ),
                    }
                ]
            )

            record.to_csv(
                self.log_file,
                mode="a",
                index=False,
                header=False,
            )

            logger.info(
                "Prediction logged successfully."
            )

        except Exception as e:

            logger.exception(
                "Failed to log prediction."
            )

            raise AutoMLException(e)

    def load_logs(self) -> pd.DataFrame:
        """
        Load monitoring logs.
        """

        try:

            if not self.log_file.exists():

                return pd.DataFrame()

            return pd.read_csv(
                self.log_file,
            )

        except Exception as e:

            logger.exception(
                "Failed to load monitoring logs."
            )

            raise AutoMLException(e)

    def summary(self) -> dict:
        """
        Return monitoring summary.
        """

        try:

            df = self.load_logs()

            if df.empty:

                return {
                    "Total Predictions": 0,
                    "Average Inference Time": 0.0,
                    "Average Probability": 0.0,
                }

            probability = (
                round(
                    df["Probability"].mean(),
                    4,
                )
                if "Probability" in df.columns
                else 0.0
            )

            return {
                "Total Predictions": len(df),
                "Average Inference Time": round(
                    df["Inference Time (ms)"].mean(),
                    2,
                ),
                "Average Probability": probability,
            }

        except Exception as e:

            logger.exception(
                "Failed to generate monitoring summary."
            )

            raise AutoMLException(e)