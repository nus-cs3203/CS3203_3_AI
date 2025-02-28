import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict

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

    def __init__(self, column_name: str, allowed_formats: List[str], logger: ValidatorLogger, custom_formats: Dict[str, callable] = None) -> None:
        """
        :param column_name: The column in the DataFrame to validate.
        :param allowed_formats: List of allowed formats (e.g., ['json', 'numpy', 'txt']).
        :param logger: Logger instance for logging validation results.
        :param custom_formats: Optional dictionary of additional format validation functions.
        """
        BaseValidationHandler.__init__(self)  # Explicitly call base class init
        self.column_name = column_name
        self.allowed_formats = set(allowed_formats)
        self.logger = logger

        # Merge default format checks with custom ones if provided
        self.format_checks = {**self.FORMAT_CHECKS, **(custom_formats or {})}

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that each value in the specified DataFrame column matches one of the allowed formats.
        Logs errors and drops invalid rows.
        """
        self.logger.log_dataframe(df)
        
        if self.column_name not in df.columns:
            error_message = f"Validation failed: Column '{self.column_name}' does not exist in DataFrame."
            self.logger.log_failure(self.column_name, error_message)
            return {"error": error_message}
        
        invalid_indices = []
        for index, value in df[self.column_name].items():
            if not any(check(value) for fmt, check in self.format_checks.items() if fmt in self.allowed_formats):
                error_message = f"Row {index}: '{self.column_name}' must be one of {self.allowed_formats}"
                self.logger.log_failure(self.column_name, error_message)
                invalid_indices.append(index)
        
        # Drop invalid rows
        if invalid_indices:
            df = df.drop(index=invalid_indices).reset_index(drop=True)
            self.logger.log_failure(self.column_name, f"Dropped {len(invalid_indices)} invalid rows.")
        else:
            self.logger.log_success(self.column_name)

        return self._validate_next(df)  # Pass to the next handler in the chain

    @staticmethod
    def is_json_serializable(value) -> bool:
        """ Helper function to check if a dictionary or list is JSON serializable """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
