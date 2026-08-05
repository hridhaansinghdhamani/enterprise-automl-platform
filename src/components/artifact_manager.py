"""
Enterprise Artifact Manager

Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import os

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger
from src.utils.common import save_object

logger = get_logger(__name__)


class ArtifactManager:
    """
    Manage saving of trained models
    and preprocessing artifacts.
    """

    @staticmethod
    def save_model(
        path: str,
        model,
    ) -> None:
        """
        Save trained model.
        """

        try:

            os.makedirs(
                os.path.dirname(path),
                exist_ok=True,
            )

            save_object(
                path,
                model,
            )

            logger.info(
                "Model saved successfully: %s",
                path,
            )

        except Exception as e:

            logger.exception(
                "Failed to save model."
            )

            raise AutoMLException(e)

    @staticmethod
    def save_preprocessor(
        path: str,
        preprocessor,
    ) -> None:
        """
        Save preprocessing pipeline.
        """

        try:

            os.makedirs(
                os.path.dirname(path),
                exist_ok=True,
            )

            save_object(
                path,
                preprocessor,
            )

            logger.info(
                "Preprocessor saved successfully: %s",
                path,
            )

        except Exception as e:

            logger.exception(
                "Failed to save preprocessor."
            )

            raise AutoMLException(e)