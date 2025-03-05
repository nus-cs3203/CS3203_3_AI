import pandas as pd
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger
from typing import Dict, Tuple, Optional

class LengthValidator(BaseValidationHandler):
    """
    Validates that specified text columns have string lengths within the given range.
    """

    def __init__(self, text_cols: Dict[str, Tuple[int, int]], logger: ValidatorLogger, log_valid: bool = False) -> None:
        """
        :param text_cols: Dictionary where keys are column names, and values are (min_length, max_length) tuples.
        :param logger: Logger instance.
        :param log_valid: Whether to log successful validations (default: False).
        """
        super().__init__()
        self.text_cols = text_cols  # e.g., {"name": (3, 50), "description": (10, 200)}
        self.logger = logger
        self.log_valid = log_valid  # Reduce log spam by default
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
        Validates the length of text columns.
        :param df: pandas DataFrame.
        :return: Dictionary with success status and error details if validation fails.
        """
        self.logger.log_dataframe(df)
        invalid_indices = set()
        errors = []

        for col, (min_length, max_length) in self.text_cols.items():
            if col not in df.columns:
                warning_message = f"Warning: Column '{col}' not found in DataFrame. Skipping validation."
                self.logger.log_warning(col, warning_message)
                continue

            # Ensure valid (non-NaN) values are treated as strings and stripped
            valid_mask = df[col].notna()
            valid_texts = df.loc[valid_mask, col].astype(str).str.strip()
            
            # Ensure the dtype is string before measuring length
            text_lengths = valid_texts[valid_texts != ""].str.len()

            if text_lengths.empty:
                continue  # If no valid strings exist, skip validation
            
            # Identify invalid rows
            invalid_rows = text_lengths[~text_lengths.between(min_length, max_length)].index.tolist()
            invalid_indices.update(invalid_rows)

            if invalid_rows:
                error_message = f"Column '{col}' must have length between {min_length} and {max_length}. Found {len(invalid_rows)} invalid rows."
                self.logger.log_failure(col, error_message)
                errors.append(error_message)
                self.is_valid = False
            elif self.log_valid:
                self.logger.log_success(col)

        validation_result = {"success": self.is_valid, "errors": errors, "invalid_indices": list(invalid_indices)}

        if not self.is_valid:
            raise ValueError(f"Validation failed. Errors: {errors}")
        
        # Pass the validation result to the next handler if valid
        if self._next_handler:
            next_result = self._next_handler.validate(df)
            validation_result["errors"].extend(next_result.get("errors", []))
            validation_result["success"] = validation_result["success"] and next_result["success"]
            validation_result["invalid_indices"].extend(next_result.get("invalid_indices", []))

        return validation_result
