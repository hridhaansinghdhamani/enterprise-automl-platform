import pandas as pd

from src.logger.logger import get_logger
from src.exception.exception import AutoMLException
from src.utils.common import read_yaml

logger = get_logger(__name__)


class DataValidation:

    def __init__(self, config):
        self.config = config

    def validate(self, dataframe: pd.DataFrame):

        try:

            schema = read_yaml(self.config.schema_path)

            expected_columns = list(schema["columns"].keys())

            missing_columns = [
                col
                for col in expected_columns
                if col not in dataframe.columns
            ]

            extra_columns = [
                col
                for col in dataframe.columns
                if col not in expected_columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing Columns : {missing_columns}"
                )

            if extra_columns:

                logger.warning(
                    f"Extra Columns Found : {extra_columns}"
                )

            logger.info("Data validation completed successfully.")

            return True

        except Exception as e:

            logger.exception("Data validation failed.")

            raise AutoMLException(
                e,
                __import__("sys"),
            )