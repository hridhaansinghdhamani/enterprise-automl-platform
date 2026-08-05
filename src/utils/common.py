"""
Enterprise Common Utility Functions
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


def create_directories(paths: list[str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_object(file_path: str, obj: Any) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path: str) -> Any:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    return joblib.load(file_path)


def read_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    return pd.read_csv(file_path)


def save_csv(df: pd.DataFrame, file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def read_yaml(file_path: str) -> dict:
    """
    Read YAML configuration file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_yaml(file_path: str, data: dict) -> None:
    """
    Write YAML configuration file.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_file_size(file_path: str) -> str:
    size = os.path.getsize(file_path)
    return f"{round(size / 1024, 2)} KB"