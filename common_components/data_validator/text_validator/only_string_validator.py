import pandas as pd
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger
from typing import List, Optional

class OnlyStringValidator(BaseValidationHandler):
    """
    Validates that specified fields in a DataFrame contain only string values.
    """

    def __init__(self, text_cols: List[str], logger: ValidatorLogger, allow_nan: bool = True) -> None:
        """
        :param text_cols: List of column names that should contain only strings.
        :param logger: Logger instance.
        :param allow_nan: Whether NaN values should be treated as valid (default: True).
        """
        super().__init__()
        self.text_cols = text_cols  # Example: ["name", "description"]
        self.logger = logger
        self.allow_nan = allow_nan  # Controls NaN validation behavior
        self.is_valid = True
        self._next_handler: Optional[BaseValidationHandler] = None  # Next validator in the chain

    def set_next(self, handler: BaseValidationHandler) -> BaseValidationHandler:
        """
        Sets the next handler in the chain, only if validation has passed.
        """
        if not self.is_valid:
            raise ValueError("Cannot set next handler because the current validation failed.")
        self._next_handler = handler
        return handler

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the specified text columns contain only string values.
        :param df: pandas DataFrame.
        :return: Dictionary with success status and error details.
        """
        self.logger.log_dataframe(df)
        errors = []

        for col in self.text_cols:
            if col not in df.columns:
                warning_message = f"Warning: Column '{col}' not found in DataFrame. Skipping validation."
                self.logger.log_warning(col, warning_message)
                continue

            # Vectorized check for non-string values
            invalid_mask = ~df[col].map(lambda x: isinstance(x, str) or (self.allow_nan and pd.isna(x)))
            invalid_rows = df.index[invalid_mask].tolist()

            if invalid_rows:
                error_message = f"Column '{col}' must contain only strings. Found {len(invalid_rows)} invalid rows."
                self.logger.log_failure(col, error_message)
                errors.append(error_message)
                self.is_valid = False
            else:
                self.logger.log_success(col)

        validation_result = {"success": self.is_valid, "errors": errors}

        if not self.is_valid:
            raise ValueError(f"Validation failed. Errors: {errors}")
        
        # Pass to next handler if exists and no errors
        if self._next_handler:
            next_result = self._next_handler.validate(df)
            validation_result["errors"].extend(next_result.get("errors", []))
            validation_result["success"] = validation_result["success"] and next_result["success"]

        return validation_result
