import re
import pandas as pd
from common_components.data_validator.validation_handler import ValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class RegexValidator(ValidationHandler):
    """
    Validates that specified string fields in a DataFrame match a given regex pattern.
    """

    def __init__(self, columns: list[str], patterns: list[str], logger: ValidatorLogger, allow_nan: bool = True, log_sample: int = 3) -> None:
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
        self.patterns = [re.compile(pattern) for pattern in patterns]
        self.logger = logger
        self.allow_nan = allow_nan
        self.log_sample = log_sample  # Controls sample size for error logging

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the specified columns in the DataFrame match their respective regex patterns.
        :param df: pandas DataFrame.
        :return: Dictionary with success status and error details if validation fails.
        """
        self.logger.log_dataframe(df)  # Log the incoming DataFrame
        errors = []

        for col, pattern in zip(self.columns, self.patterns):
            if col not in df.columns:
                warning_message = f"Warning: Column '{col}' not found in DataFrame. Skipping validation."
                self.logger.log_failure(col, warning_message)
                continue  # Skip missing columns

            # Handle NaN values based on configuration
            if self.allow_nan:
                invalid_mask = ~df[col].astype(str).str.match(pattern, na=True)
            else:
                invalid_mask = ~df[col].astype(str).str.match(pattern, na=False)

            invalid_rows = df[invalid_mask].index.tolist()

            if invalid_rows:
                sample_invalid_values = df.loc[invalid_rows, col].head(self.log_sample).tolist()
                error_message = f"Column '{col}' contains values that do not match the pattern. Example: {sample_invalid_values}"
                self.logger.log_failure(col, error_message)
                errors.append({col: {"invalid_rows": invalid_rows, "error": error_message}})
            else:
                self.logger.log_success(col)

        if errors:
            return {"success": False, "errors": errors}

        return self._validate_next(df)
