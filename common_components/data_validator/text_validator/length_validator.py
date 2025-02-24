import pandas as pd
from common_components.data_validator.validation_handler import ValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class LengthValidator(ValidationHandler):
    """
    Validates that specified text columns have string lengths within the given range.
    """

    def __init__(self, text_cols: dict, logger: ValidatorLogger, log_valid: bool = False) -> None:
        """
        :param text_cols: Dictionary where keys are column names, and values are (min_length, max_length) tuples.
        :param logger: Logger instance.
        :param log_valid: Whether to log successful validations (default: False).
        """
        super().__init__()
        self.text_cols = text_cols  # e.g., {"name": (3, 50), "description": (10, 200)}
        self.logger = logger
        self.log_valid = log_valid  # Reduce log spam by default

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates the length of text columns.
        :param df: pandas DataFrame.
        :return: Dictionary with success status and error details if validation fails.
        """
        self.logger.log_dataframe(df)  # Log the incoming DataFrame
        errors = []

        for col, (min_length, max_length) in self.text_cols.items():
            if col not in df.columns:
                warning_message = f"Warning: Column '{col}' not found in DataFrame. Skipping validation."
                self.logger.log_warning(col, warning_message)
                continue  # Skip missing columns

            # Ensure NaN values are treated safely
            valid_mask = df[col].notna()
            text_lengths = df[col].astype(str).str.len()

            # Identify invalid rows
            invalid_rows = df[valid_mask & ~text_lengths.between(min_length, max_length)].index.tolist()

            if invalid_rows:
                error_message = f"Column '{col}' must have length between {min_length} and {max_length}."
                self.logger.log_failure(col, error_message)
                errors.append({col: {"invalid_rows": invalid_rows, "error": error_message}})
            elif self.log_valid:
                self.logger.log_success(col)

        if errors:
            return {"success": False, "errors": errors}

        return self._validate_next(df)
