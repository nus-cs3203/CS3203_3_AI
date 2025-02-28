import re
import pandas as pd
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger
from typing import List, Optional

class RegexValidator(BaseValidationHandler):
    """
    Validates that specified string fields in a DataFrame match a given regex pattern.
    """

    def __init__(self, columns: List[str], patterns: List[str], logger: ValidatorLogger, allow_nan: bool = True, log_sample: int = 3) -> None:
        """
        :param columns: List of column names to validate.
        :param patterns: List of corresponding regex patterns (must be same length as `columns`).
        :param logger: Logger instance.
        :param allow_nan: Whether NaN values should be treated as valid (default: True).
        :param log_sample: Number of sample invalid values to log for debugging.
        """
        super().__init__()

        if len(columns) != len(patterns):
            raise ValueError("`columns` and `patterns` must have the same length.")

        self.columns = columns
        self.patterns = patterns  # Store as raw strings instead of compiled regex
        self.logger = logger
        self.allow_nan = allow_nan
        self.log_sample = log_sample
        self.next_handler: Optional[BaseValidationHandler] = None

    def set_next(self, handler: 'BaseValidationHandler') -> 'BaseValidationHandler':
        """
        Sets the next handler in the chain.
        """
        self.next_handler = handler
        return handler

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the specified columns in the DataFrame match their respective regex patterns.
        :param df: pandas DataFrame.
        :return: Dictionary with success status and error details.
        """
        self.logger.log_dataframe(df)
        invalid_indices = set()
        errors = []

        for col, pattern in zip(self.columns, self.patterns):
            if col not in df.columns:
                warning_message = f"Warning: Column '{col}' not found in DataFrame. Skipping validation."
                self.logger.log_warning(col, warning_message)
                continue

            # Apply regex validation using raw pattern string
            regex = re.compile(pattern)
            valid_mask = df[col].astype(str).str.match(regex, na=self.allow_nan)
            invalid_rows = df.loc[~valid_mask].index.tolist()
            invalid_indices.update(invalid_rows)

            if invalid_rows:
                sample_invalid_values = df.loc[invalid_rows, col].head(self.log_sample).tolist()
                error_message = (
                    f"Column '{col}' contains invalid values. "
                    f"Examples: {sample_invalid_values} (Dropping {len(invalid_rows)} rows)"
                )
                self.logger.log_failure(col, error_message)
                errors.append(error_message)
            else:
                self.logger.log_success(col)

        # Drop invalid rows
        if invalid_indices:
            df = df.drop(index=list(invalid_indices)).reset_index(drop=True)

        validation_result = {"success": not bool(errors), "errors": errors}

        # Pass to next handler if exists
        if self.next_handler:
            next_result = self.next_handler.validate(df)
            validation_result["errors"].extend(next_result.get("errors", []))
            validation_result["success"] = validation_result["success"] and next_result["success"]

        return validation_result
