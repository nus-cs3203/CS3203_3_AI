import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Optional, Callable

from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class FormatValidator(BaseValidationHandler):
    """
    Concrete Handler that validates if a specified DataFrame column matches an allowed format.
    Supported formats: JSON (dict or list of dicts), NumPy array (or list of arrays), or .txt file paths.
    """

    FORMAT_CHECKS = {
        "json": lambda value: isinstance(value, (dict, list)) and FormatValidator.is_json_serializable(value),
        "numpy": lambda value: isinstance(value, (np.ndarray, list)) and all(
            isinstance(v, np.ndarray) for v in (value if isinstance(value, list) else [value])
        ),
        "txt": lambda value: isinstance(value, str) and value.lower().endswith('.txt') and os.path.isfile(value)
    }

    def __init__(self, column_name: str, allowed_formats: List[str], logger: ValidatorLogger, custom_formats: Optional[Dict[str, Callable]] = None) -> None:
        """
        :param column_name: The column in the DataFrame to validate.
        :param allowed_formats: List of allowed formats (e.g., ['json', 'numpy', 'txt']).
        :param logger: Logger instance for logging validation results.
        :param custom_formats: Optional dictionary of additional format validation functions.
        """
        super().__init__()
        self.column_name = column_name
        self.allowed_formats = set(allowed_formats)
        self.logger = logger
        self.format_checks = {**self.FORMAT_CHECKS, **(custom_formats or {})}
        self._next_handler: Optional[BaseValidationHandler] = None  # Only set if validation passes

    def set_next(self, handler: 'BaseValidationHandler') -> 'BaseValidationHandler':
        """
        Sets the next handler in the chain only if this validator has not failed.
        """
        self._next_handler = handler
        return handler

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that each value in the specified DataFrame column matches one of the allowed formats.
        Logs errors and raises a ValueError if errors are found.
        """
        self.logger.log_dataframe(df)

        if self.column_name not in df.columns:
            error_message = f"Validation failed: Column '{self.column_name}' does not exist in DataFrame."
            self.logger.log_failure(self.column_name, error_message)
            raise ValueError(error_message)

        # Track errors for invalid values
        errors = []
        for index, value in df[self.column_name].items():
            if not any(self.format_checks[fmt](value) for fmt in self.allowed_formats):
                error_message = f"Row {index}: '{self.column_name}' must be one of {self.allowed_formats}"
                self.logger.log_failure(self.column_name, error_message)
                errors.append(error_message)

        if errors:
            # Raise ValueError with all the collected errors
            raise ValueError(f"Validation failed for column '{self.column_name}':\n" + "\n".join(errors))

        # Log success if no errors
        self.logger.log_success(self.column_name)

        # Pass validation to the next handler if valid
        return self._validate_next(df)

    @staticmethod
    def is_json_serializable(value) -> bool:
        """ Helper function to check if a dictionary or list is JSON serializable """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

    def _validate_next(self, df: pd.DataFrame) -> dict:
        """
        Pass the DataFrame to the next handler in the chain if it exists.
        """
        if self._next_handler:
            return self._next_handler.validate(df)

        return {"success"}
