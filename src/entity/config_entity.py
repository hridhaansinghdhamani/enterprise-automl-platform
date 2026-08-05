from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    root_dir: str
    train_path: str
    test_path: str


@dataclass
class DataValidationConfig:
    root_dir: str
    status_file: str
    schema_path: str


@dataclass
class DataTransformationConfig:
    root_dir: str
    preprocessor_path: str
    train_array_path: str
    test_array_path: str


@dataclass
class ModelTrainerConfig:
    root_dir: str
    model_path: str
    leaderboard_path: str


@dataclass
class ModelEvaluationConfig:
    root_dir: str


@dataclass
class PredictionConfig:
    root_dir: str