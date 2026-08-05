from dataclasses import dataclass

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from xgboost import XGBClassifier, XGBRegressor

from src.exception.exception import AutoMLException
from src.logger.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    name: str
    model: object
    task: str
    probability: bool
    feature_importance: bool
    optuna: bool


class ModelFactory:
    """
    Factory class to create classification
    and regression models.
    """

    def get_classification_models(self):

        try:

            logger.info(
                "Loading classification models..."
            )

            models = [
                ModelInfo(
                    name="Logistic Regression",
                    model=LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                    ),
                    task="classification",
                    probability=True,
                    feature_importance=False,
                    optuna=False,
                ),
                ModelInfo(
                    name="Random Forest",
                    model=RandomForestClassifier(
                        random_state=42,
                        n_jobs=-1,
                    ),
                    task="classification",
                    probability=True,
                    feature_importance=True,
                    optuna=True,
                ),
                ModelInfo(
                    name="XGBoost",
                    model=XGBClassifier(
                        random_state=42,
                        eval_metric="logloss",
                        verbosity=0,
                        n_jobs=-1,
                    ),
                    task="classification",
                    probability=True,
                    feature_importance=True,
                    optuna=True,
                ),
                ModelInfo(
                    name="LightGBM",
                    model=LGBMClassifier(
                        random_state=42,
                        verbose=-1,
                    ),
                    task="classification",
                    probability=True,
                    feature_importance=True,
                    optuna=True,
                ),
                ModelInfo(
                    name="CatBoost",
                    model=CatBoostClassifier(
                        random_state=42,
                        verbose=False,
                    ),
                    task="classification",
                    probability=True,
                    feature_importance=True,
                    optuna=True,
                ),
            ]

            logger.info(
                "%d classification models loaded.",
                len(models),
            )

            return models

        except Exception as e:

            logger.exception(
                "Failed to load classification models."
            )

            raise AutoMLException(e)

    def get_regression_models(self):

        try:

            logger.info(
                "Loading regression models..."
            )

            models = [
                ModelInfo(
                    name="Linear Regression",
                    model=LinearRegression(),
                    task="regression",
                    probability=False,
                    feature_importance=False,
                    optuna=False,
                ),
                ModelInfo(
                    name="Random Forest",
                    model=RandomForestRegressor(
                        random_state=42,
                        n_jobs=-1,
                    ),
                    task="regression",
                    probability=False,
                    feature_importance=True,
                    optuna=True,
                ),
                ModelInfo(
                    name="XGBoost",
                    model=XGBRegressor(
                        random_state=42,
                        verbosity=0,
                        n_jobs=-1,
                    ),
                    task="regression",
                    probability=False,
                    feature_importance=True,
                    optuna=True,
                ),
                ModelInfo(
                    name="LightGBM",
                    model=LGBMRegressor(
                        random_state=42,
                        verbose=-1,
                    ),
                    task="regression",
                    probability=False,
                    feature_importance=True,
                    optuna=True,
                ),
                ModelInfo(
                    name="CatBoost",
                    model=CatBoostRegressor(
                        random_state=42,
                        verbose=False,
                    ),
                    task="regression",
                    probability=False,
                    feature_importance=True,
                    optuna=True,
                ),
            ]

            logger.info(
                "%d regression models loaded.",
                len(models),
            )

            return models

        except Exception as e:

            logger.exception(
                "Failed to load regression models."
            )

            raise AutoMLException(e)