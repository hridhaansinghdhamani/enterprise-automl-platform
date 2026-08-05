"""
Enterprise Model Trainer
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import os
import time

import pandas as pd
from sklearn.model_selection import cross_val_score

from src.components.metrics_engine import MetricsEngine
from src.components.model_evaluation import ModelEvaluation
from src.components.model_factory import ModelFactory
from src.components.model_registry import ModelRegistry
from src.components.optuna_tuner import OptunaTuner
from src.exception.exception import AutoMLException
from src.logger.logger import get_logger
from src.utils.common import save_object

logger = get_logger(__name__)


class ModelTrainer:
    """
    Enterprise Model Trainer
    """

    def __init__(self, config):

        self.config = config

        self.factory = ModelFactory()

        self.optimizer = OptunaTuner()

        self.metrics = MetricsEngine()

        self.registry = ModelRegistry()

        self.evaluator = ModelEvaluation()

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        task,
        cv: int = 5,
        n_trials: int = 30,
    ):

        try:

            if task == "classification":

                models = (
                    self.factory.get_classification_models()
                )

                scoring = "accuracy"

            else:

                models = (
                    self.factory.get_regression_models()
                )

                scoring = "r2"

            leaderboard = []

            best_model = None

            best_name = None

            best_score = float("-inf")

            for model_info in models:

                logger.info(
                    "Training %s",
                    model_info.name,
                )

                model = self.optimizer.optimize(
                    model_info,
                    X_train,
                    y_train,
                    task=task,
                    cv=cv,
                    n_trials=n_trials,
                )

                start = time.time()

                score = cross_val_score(
                    estimator=model,
                    X=X_train,
                    y=y_train,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                ).mean()

                model.fit(
                    X_train,
                    y_train,
                )

                prediction = model.predict(
                    X_test,
                )

                training_time = round(
                    time.time() - start,
                    2,
                )

                if task == "classification":

                    probability = None

                    if hasattr(
                        model,
                        "predict_proba",
                    ):

                        try:

                            probability = (
                                model.predict_proba(
                                    X_test
                                )[:, 1]
                            )

                        except Exception:

                            probability = None

                    metrics = (
                        self.metrics.classification(
                            y_test,
                            prediction,
                            probability,
                        )
                    )

                else:

                    metrics = (
                        self.metrics.regression(
                            y_test,
                            prediction,
                        )
                    )

                leaderboard.append(
                    {
                        "Model": model_info.name,
                        "CV Score": round(
                            score,
                            4,
                        ),
                        "Training Time (s)": training_time,
                        **metrics,
                    }
                )

                if score > best_score:

                    best_score = score

                    best_model = model

                    best_name = model_info.name

            leaderboard = (
                pd.DataFrame(
                    leaderboard
                )
                .sort_values(
                    by="CV Score",
                    ascending=False,
                )
                .reset_index(
                    drop=True,
                )
            )

            # ============================
            # Create required directories
            # ============================

            trainer_dir = (
                self.config.root_dir
            )

            os.makedirs(
                trainer_dir,
                exist_ok=True,
            )

            os.makedirs(
                os.path.dirname(
                    self.config.model_path
                ),
                exist_ok=True,
            )

            os.makedirs(
                os.path.dirname(
                    self.config.leaderboard_path
                ),
                exist_ok=True,
            )

            logger.info(
                "Saving leaderboard..."
            )

            leaderboard.to_csv(
                self.config.leaderboard_path,
                index=False,
            )

            logger.info(
                "Saving best model..."
            )

            save_object(
                self.config.model_path,
                best_model,
            )

            logger.info(
                "Registering model..."
            )

            self.registry.register(
                self.config.model_path,
                best_name,
            )

            logger.info("=" * 60)

            logger.info(
                "Best Model : %s",
                best_name,
            )

            logger.info(
                "Best CV Score : %.4f",
                best_score,
            )

            logger.info("=" * 60)

            return {
                "model": best_model,
                "leaderboard": leaderboard,
                "best_model": best_name,
                "best_score": best_score,
                "metrics": leaderboard.iloc[
                    0
                ].to_dict(),
            }

        except Exception as e:

            logger.exception(
                "Model training failed."
            )

            raise AutoMLException(e)