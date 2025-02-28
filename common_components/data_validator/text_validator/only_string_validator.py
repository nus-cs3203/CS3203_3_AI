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
        self.next_handler: Optional[BaseValidationHandler] = None  # Next validator in the chain

    def set_next(self, handler: 'BaseValidationHandler') -> 'BaseValidationHandler':
        """
        Sets the next handler in the chain.
        """
        self.next_handler = handler
        return handler

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the specified text columns contain only string values.
        :param df: pandas DataFrame.
        :return: Dictionary with success status and error details.
        """
        self.logger.log_dataframe(df)
        invalid_indices = set()
        errors = []

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
