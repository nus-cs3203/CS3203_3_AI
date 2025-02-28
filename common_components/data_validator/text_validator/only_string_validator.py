import pandas as pd
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class OnlyStringValidator(BaseValidationHandler):
    """
    Validates that specified fields in a DataFrame contain only string values.
    """

    def __init__(self, text_cols: list, logger: ValidatorLogger, allow_nan: bool = True) -> None:
        """
        :param text_cols: List of column names that should contain only strings.
        :param logger: Logger instance.
        :param allow_nan: Whether NaN values should be treated as valid (default: True).
        """
        BaseValidationHandler.__init__(self)  # Explicitly call Base class init
        self.text_cols = text_cols  # Example: ["name", "description"]
        self.logger = logger
        self.allow_nan = allow_nan  # Controls NaN validation behavior

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the specified text columns contain only string values.
        :param df: pandas DataFrame.
        :return: Updated DataFrame after dropping invalid rows.
        """
        self.logger.log_dataframe(df)  # Log incoming DataFrame
        invalid_indices = set()

        for col in self.text_cols:
            if col not in df.columns:
                warning_message = f"Warning: Column '{col}' not found in DataFrame. Skipping validation."
                self.logger.log_warning(col, warning_message)
                continue  # Skip missing columns

            # Identify non-string values
            valid_mask = df[col].apply(lambda x: isinstance(x, str) or (self.allow_nan and pd.isna(x)))
            invalid_rows = df.loc[~valid_mask].index.tolist()
            invalid_indices.update(invalid_rows)

            if invalid_rows:
                error_message = f"Column '{col}' must contain only strings. Dropping {len(invalid_rows)} rows."
                self.logger.log_failure(col, error_message)
            else:
                self.logger.log_success(col)

        # Drop invalid rows
        if invalid_indices:
            df = df.drop(index=list(invalid_indices)).reset_index(drop=True)

        return self._validate_next(df)  # Pass to next validator
