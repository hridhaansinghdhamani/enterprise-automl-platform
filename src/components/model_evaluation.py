from sklearn.metrics import *

from src.components.metrics_engine import MetricsEngine


class ModelEvaluation:

    def evaluate(

        self,

        model,

        X_test,

        y_test,

        task,

    ):

        prediction = model.predict(X_test)

        metric = MetricsEngine()

        if task == "classification":

            return metric.classification(

                y_test,

                prediction,

            )

        return metric.regression(

            y_test,

            prediction,

        )