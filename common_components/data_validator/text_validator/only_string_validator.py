import pandas as pd
from common_components.data_validator.validation_handler import ValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class OnlyStringValidator(ValidationHandler):
    """
    Validates that specified fields in a DataFrame contain only string values.
    """

    def __init__(self, text_cols: list, logger: ValidatorLogger, allow_nan: bool = True) -> None:
        """
        :param text_cols: List of column names that should contain only strings.
        :param logger: Logger instance.
        :param allow_nan: Whether NaN values should be treated as valid (default: True).
        """
        super().__init__()
        self.text_cols = text_cols  # Example: ["name", "description"]
        self.logger = logger
        self.allow_nan = allow_nan  # Controls NaN validation behavior

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the specified text columns contain only string values.
        :param df: pandas DataFrame.
        :return: Dictionary with success status and error details if validation fails.
        """
        self.logger.log_dataframe(df)  # Log the incoming DataFrame
        errors = []

        for col in self.text_cols:
            if col not in df.columns:
                warning_message = f"Warning: Column '{col}' not found in DataFrame. Skipping validation."
                self.logger.log_warning(col, warning_message)
                continue  # Skip missing columns

            # Handle NaN values based on configuration
            if self.allow_nan:
                invalid_mask = ~df[col].apply(lambda x: isinstance(x, str) or pd.isna(x))
            else:
                invalid_mask = ~df[col].apply(lambda x: isinstance(x, str))

            invalid_rows = df[invalid_mask].index.tolist()

            if invalid_rows:
                error_message = f"Column '{col}' must contain only strings."
                self.logger.log_failure(col, error_message)
                errors.append({col: {"invalid_rows": invalid_rows, "error": error_message}})
            else:
                self.logger.log_success(col)

        if errors:
            return {"success": False, "errors": errors}

        return self._validate_next(df)
