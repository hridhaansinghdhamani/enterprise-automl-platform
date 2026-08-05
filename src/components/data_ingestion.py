import pandas as pd

from src.logger.logger import logger
from src.exception.exception import AutoMLException


class DataIngestion:

    def __init__(self, config):
        self.config = config

    def load_data(self, file_path):
        try:
            logger.info("Loading dataset")

            df = pd.read_csv(file_path)

            logger.info(f"Dataset Loaded : {df.shape}")

            return df

        except Exception as e:
            raise AutoMLException(e, __import__("sys"))