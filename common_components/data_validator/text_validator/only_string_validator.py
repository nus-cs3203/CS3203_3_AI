import pandas as pd
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger
from typing import List, Optional

class OnlyStringValidator(BaseValidationHandler):
    """
    Validates that specified fields in a DataFrame contain only string values.
    """

    def __init__(self, text_cols: List[str], logger: Optional[ValidatorLogger] = None, allow_nan: bool = True) -> None:
        """
        :param text_cols: List of column names that should contain only strings.
        :param logger: Optional logger instance. Uses default logger if not provided.
        :param allow_nan: Whether NaN values should be treated as valid (default: True).
        """
        super().__init__()
        self.text_cols = text_cols
        self.logger = logger or ValidatorLogger()
        self.allow_nan = allow_nan
        self._next_handler: Optional[BaseValidationHandler] = None

    def set_next(self, handler: BaseValidationHandler) -> BaseValidationHandler:
        """
        Set the next handler in the chain only if the current validation has passed.
        """
        self._next_handler = handler
        return handler

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the specified text columns contain only string values.
        :param df: pandas DataFrame.
        :return: Dictionary with success status and error details.
        """
        self.logger.log_dataframe(df)

        missing_cols = [col for col in self.text_cols if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                self.logger.log_failure(col, f"Validation failed: Column '{col}' not found in DataFrame.")
            raise ValueError("Current validation failed due to missing columns.")

        errors = []
        for col in self.text_cols:
            invalid_mask = ~df[col].map(lambda x: isinstance(x, str) or (self.allow_nan and pd.isna(x)))
            invalid_rows = df.index[invalid_mask].tolist()

            if invalid_rows:
                error_message = f"Column '{col}' must contain only strings. Found {len(invalid_rows)} invalid rows."
                self.logger.log_failure(col, error_message)
                errors.append(error_message)
                self.is_valid = False
            else:
                self.logger.log_success(col)

        if errors:
            raise ValueError(f"Current validation failed due to invalid rows: {errors}")

        # Pass to next handler if it exists
        return self._validate_next(df)

    def _validate_next(self, df: pd.DataFrame) -> dict:
        """
        Pass the DataFrame to the next handler in the chain if it exists.
        """
        if self._next_handler:
            return self._next_handler.validate(df)

        return {"success": True}
