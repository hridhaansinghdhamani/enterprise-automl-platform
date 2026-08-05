from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

CONFIGS_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

CLASSIFICATION_THRESHOLD = 10
IMBALANCE_THRESHOLD = 0.20