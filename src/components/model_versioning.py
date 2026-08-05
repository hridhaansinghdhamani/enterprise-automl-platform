"""
Enterprise Model Versioning

Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


class ModelVersioning:
    """
    Automatically versions trained models.

    Example
    -------
    saved_models/
        latest.pkl
        model_v1.pkl
        model_v2.pkl
        metadata.json
    """

    def __init__(
        self,
        model_dir: str = "saved_models",
    ) -> None:

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_path = (
            self.model_dir / "metadata.json"
        )

        if not self.metadata_path.exists():

            self._save_metadata(
                {
                    "latest_version": 0,
                    "models": [],
                }
            )

    def _load_metadata(self) -> dict:

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    def _save_metadata(
        self,
        metadata: dict,
    ) -> None:

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

    def register(
        self,
        model_path: str,
    ) -> str:
        """
        Register a new model version.

        Returns
        -------
        str
            Path of newly registered model.
        """

        try:

            source = Path(model_path)

            if not source.exists():
                raise FileNotFoundError(
                    f"Model not found: {source}"
                )

            metadata = self._load_metadata()

            version = (
                metadata["latest_version"] + 1
            )

            destination = (
                self.model_dir
                / f"model_v{version}.pkl"
            )

            shutil.copy2(
                source,
                destination,
            )

            shutil.copy2(
                source,
                self.model_dir / "latest.pkl",
            )

            metadata["latest_version"] = version

            metadata["models"].append(
                {
                    "version": version,
                    "file": destination.name,
                    "created_at": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )

            self._save_metadata(metadata)

            logger.info(
                "Model version %d registered successfully.",
                version,
            )

            return str(destination)

        except Exception as e:

            logger.exception(
                "Model versioning failed."
            )

            raise AutoMLException(e)

    def latest_model(self) -> str:
        """
        Return latest model path.
        """

        return str(
            self.model_dir / "latest.pkl"
        )

    def list_versions(self) -> list:
        """
        Return registered model versions.
        """

        return self._load_metadata().get(
            "models",
            [],
        )