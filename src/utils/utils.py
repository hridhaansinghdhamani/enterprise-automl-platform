"""
General Utility Functions
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: str) -> dict:
    """
    Load YAML configuration file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(file_path: str, data: dict) -> None:
    """
    Save dictionary as YAML.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def load_json(file_path: str) -> dict:
    """
    Load JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path: str, data: dict) -> None:
    """
    Save dictionary as JSON.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def ensure_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def file_exists(path: str) -> bool:
    """
    Check whether a file exists.
    """
    return os.path.exists(path)


def get_extension(file_path: str) -> str:
    """
    Return file extension.
    """
    return Path(file_path).suffix.lower()


def make_serializable(obj: Any):
    """
    Convert NumPy/Pandas values into JSON-serializable objects.
    """
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass

    return obj