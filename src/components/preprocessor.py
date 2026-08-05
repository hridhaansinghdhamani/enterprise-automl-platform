"""
Enterprise Preprocessor

Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Build preprocessing pipelines for numerical
    and categorical features.
    """

    def build(
        self,
        numerical_columns,
        categorical_columns,
    ) -> ColumnTransformer:
        """
        Build preprocessing pipeline.

        Parameters
        ----------
        numerical_columns : list
            Numerical feature names.

        categorical_columns : list
            Categorical feature names.

        Returns
        -------
        ColumnTransformer
        """

        try:

            logger.info(
                "Building preprocessing pipeline..."
            )

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent",
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            )

            transformer = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        numerical_pipeline,
                        numerical_columns,
                    ),
                    (
                        "cat",
                        categorical_pipeline,
                        categorical_columns,
                    ),
                ],
                remainder="drop",
            )

            logger.info(
                "Preprocessing pipeline created successfully."
            )

            return transformer

        except Exception as e:

            logger.exception(
                "Failed to build preprocessing pipeline."
            )

            raise AutoMLException(e)