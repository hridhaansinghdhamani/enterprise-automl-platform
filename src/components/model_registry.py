"""
Enterprise Model Registry

Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Register trained models into the deployment directory.
    """

    def __init__(
        self,
        registry_dir: str = "saved_models",
    ):

        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def register(
        self,
        model_path: str,
        model_name: str,
    ) -> str:
        """
        Register a trained model.

        Parameters
        ----------
        model_path : str
            Path of trained model.

        model_name : str
            Name to save inside registry.

        Returns
        -------
        str
            Registered model path.
        """

        try:

            source = Path(model_path)

            if not source.exists():
                raise FileNotFoundError(
                    f"Model file not found: {source}"
                )

            destination = (
                self.registry_dir
                / f"{model_name}.pkl"
            )

            shutil.copy2(
                source,
                destination,
            )

            logger.info(
                "Model registered successfully: %s",
                destination,
            )

            return str(destination)

        except Exception as e:

            logger.exception(
                "Failed to register model."
            )

            raise AutoMLException(e)