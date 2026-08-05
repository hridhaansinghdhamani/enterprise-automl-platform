import optuna

from sklearn.model_selection import cross_val_score


class OptunaTuner:

    def optimize(
        self,
        model_info,
        X,
        y,
        task,
        cv=5,
        n_trials=30,
    ):

        if not model_info.optuna:
            return model_info.model

        def objective(trial):

            model = model_info.model

            params = self.get_search_space(
                model_info.name,
                trial,
                task,
            )

            model.set_params(**params)

            scoring = (
                "accuracy"
                if task == "classification"
                else "r2"
            )

            score = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            ).mean()

            return score

        study = optuna.create_study(
            direction="maximize"
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
        )

        model = model_info.model

        model.set_params(
            **study.best_params
        )

        return model

    def get_search_space(
        self,
        model_name,
        trial,
        task,
    ):

        if model_name == "Random Forest":

            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    50,
                    80,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    6,
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    2,
                    10,
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    5,
                ),
            }

        elif model_name == "Extra Trees":

            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    100,
                    500,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    20,
                ),
            }

        elif model_name == "Gradient Boosting":

            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    100,
                    400,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.30,
                ),
            }

        elif model_name == "Hist Gradient Boosting":

            return {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.30,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    20,
                ),
            }

        elif model_name == "AdaBoost":

            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    50,
                    300,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    1.0,
                ),
            }

        elif model_name == "SVM":

            return {
                "C": trial.suggest_float(
                    "C",
                    0.01,
                    100,
                    log=True,
                ),
                "gamma": trial.suggest_categorical(
                    "gamma",
                    [
                        "scale",
                        "auto",
                    ],
                ),
            }

        elif model_name == "XGBoost":

            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    50,
                    80,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    5,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.30,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    0.6,
                    1.0,
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.6,
                    1.0,
                ),
            }

        elif model_name == "LightGBM":

            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    50,
                    80
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    20,
                    50,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.30,
                ),
            }

        elif model_name == "CatBoost":

            return {
                "iterations": trial.suggest_int(
                    "iterations",
                    40,
                    80,
                ),
                "depth": trial.suggest_int(
                    "depth",
                    3,
                    5,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.30,
                ),
            }

        elif model_name == "Ridge":

            return {
                "alpha": trial.suggest_float(
                    "alpha",
                    0.001,
                    100,
                    log=True,
                ),
            }

        elif model_name == "Lasso":

            return {
                "alpha": trial.suggest_float(
                    "alpha",
                    0.001,
                    100,
                    log=True,
                ),
            }

        elif model_name == "ElasticNet":

            return {
                "alpha": trial.suggest_float(
                    "alpha",
                    0.001,
                    100,
                    log=True,
                ),
                "l1_ratio": trial.suggest_float(
                    "l1_ratio",
                    0.1,
                    0.9,
                ),
            }

        elif model_name == "SVR":

            return {
                "C": trial.suggest_float(
                    "C",
                    0.01,
                    100,
                    log=True,
                ),
                "epsilon": trial.suggest_float(
                    "epsilon",
                    0.01,
                    1.0,
                ),
            }

        return {}