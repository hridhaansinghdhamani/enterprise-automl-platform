"""
Enterprise Exception Module
"""

from __future__ import annotations

import sys
import traceback
from typing import Optional


class AutoMLException(Exception):
    """
    Enterprise custom exception.
    Backward compatible with existing project.
    """

    def __init__(
        self,
        error_message: Exception | str,
        error_detail: Optional[object] = None,
    ):

        self.error_message = str(error_message)

        if (
            error_detail is not None
            and hasattr(error_detail, "exc_info")
        ):

            _, _, exc_tb = error_detail.exc_info()

            if exc_tb is not None:

                self.file_name = exc_tb.tb_frame.f_code.co_filename
                self.line_number = exc_tb.tb_lineno

            else:

                self.file_name = "Unknown"
                self.line_number = -1

        else:

            self.file_name = "Unknown"
            self.line_number = -1

        super().__init__(self.__str__())

    def __str__(self):

        return (
            f"Error occurred in [{self.file_name}] "
            f"at line [{self.line_number}] : "
            f"{self.error_message}"
        )


# Backward compatibility
CustomException = AutoMLException


def log_exception() -> str:
    """
    Returns formatted traceback.
    """

    return traceback.format_exc()