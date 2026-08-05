from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:

    train_file_path: str

    test_file_path: str


@dataclass
class DataValidationArtifact:

    validation_status: bool


@dataclass
class DataTransformationArtifact:

    train_array_path: str

    test_array_path: str

    preprocessor_path: str


@dataclass
class ModelTrainerArtifact:

    model_path: str

    leaderboard_path: str

    best_model_name: str

    best_score: float