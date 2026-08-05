"""
Enterprise Training Pipeline
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split

from src.configuration.configuration import ConfigurationManager

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.schema_generator import SchemaGenerator
from src.components.problem_detector import ProblemDetector
from src.components.preprocessor import Preprocessor
from src.components.automl_engine import AutoMLEngine
from src.components.explainability import SHAPVisualizer
from src.components.report_generator import ReportGenerator

from src.utils.common import save_object


class TrainingPipeline:

    def __init__(self):

        self.config = ConfigurationManager()

    def run_pipeline(
        self,
        file_path,
    ):

        ingestion = DataIngestion(
            self.config.get_data_ingestion_config()
        )

        dataframe = ingestion.load_data(
            file_path
        )

        DataValidation(
            self.config.get_data_validation_config()
        ).validate(
            dataframe
        )

        target_column = dataframe.columns[-1]

        SchemaGenerator().generate(
            dataframe,
            target_column,
            "configs/schema.yaml",
        )

        X = dataframe.drop(
            columns=[target_column]
        )

        y = dataframe[target_column]

        if y.dtype == "object":

            mapping = {
                "No": 0,
                "Yes": 1,
            }

            if set(
                y.unique()
            ).issubset(
                mapping.keys()
            ):

                y = y.map(
                    mapping
                )

        numerical_columns = X.select_dtypes(
            include=[
                "int64",
                "float64",
            ]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns.tolist()

        task = ProblemDetector().detect(
            y
        )

        stratify = (
            y
            if task == "classification"
            else None
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify,
        )

        preprocessor = Preprocessor().build(
            numerical_columns,
            categorical_columns,
        )

        X_train = preprocessor.fit_transform(
            X_train
        )

        X_test = preprocessor.transform(
            X_test
        )

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        X_train = pd.DataFrame(
            X_train,
            columns=feature_names,
        )

        X_test = pd.DataFrame(
            X_test,
            columns=feature_names,
        )

        save_object(
            "artifacts/data_transformation/preprocessor.pkl",
            preprocessor,
        )

            # ======================================
        # Train AutoML Models
        # ======================================

        result = AutoMLEngine(
            self.config.get_model_trainer_config()
        ).run(
            X_train,
            y_train,
            X_test,
            y_test,
        )

        # ======================================
        # Generate SHAP Explainability
        # ======================================

        try:

            SHAPVisualizer().generate(
                model=result["model"],
                X=X_train,
            )

            print(
                "SHAP report generated successfully."
            )

        except Exception as e:

            print(
                f"SHAP generation skipped: {e}"
            )    

            # ======================================
        # Generate PDF Report
        # ======================================

        try:

            ReportGenerator().generate(
                leaderboard=result["leaderboard"],
                output_path=(
                    "reports/"
                    "Enterprise_AutoML_Report.pdf"
                ),
            )

            print(
                "PDF report generated successfully."
            )

        except Exception as e:

            print(
                f"Report generation skipped: {e}"
            )

        # ======================================
        # Return Training Result
        # ======================================

        return result        