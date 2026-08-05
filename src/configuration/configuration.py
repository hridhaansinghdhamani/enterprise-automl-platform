from src.utils.common import read_yaml

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    PredictionConfig,
)


class ConfigurationManager:

    def __init__(
        self,
        config_path="configs/config.yaml",
        params_path="configs/params.yaml",
    ):

        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

    def get_data_ingestion_config(self):

        config = self.config["data_ingestion"]

        return DataIngestionConfig(
            root_dir=config["root_dir"],
            train_path=config["train_path"],
            test_path=config["test_path"],
        )

    def get_data_validation_config(self):

        config = self.config["data_validation"]

        return DataValidationConfig(
            root_dir=config["root_dir"],
            status_file=config["status_file"],
            schema_path="configs/schema.yaml",
        )

    def get_data_transformation_config(self):

        config = self.config["data_transformation"]

        return DataTransformationConfig(
            root_dir=config["root_dir"],
            preprocessor_path=config["preprocessor_path"],
            train_array_path=config["train_array_path"],
            test_array_path=config["test_array_path"],
        )

    def get_model_trainer_config(self):

        config = self.config["model_trainer"]

        return ModelTrainerConfig(
            root_dir=config["root_dir"],
            model_path=config["model_path"],
            leaderboard_path=config["leaderboard_path"],
        )

    def get_model_evaluation_config(self):

        config = self.config["model_evaluation"]

        return ModelEvaluationConfig(
            root_dir=config["root_dir"],
        )

    def get_prediction_config(self):

        config = self.config["prediction"]

        return PredictionConfig(
            root_dir=config["root_dir"],
        )