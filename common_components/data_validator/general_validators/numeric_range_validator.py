import numpy as np
import pandas as pd
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger
from typing import Optional

class NumericRangeValidator(BaseValidationHandler):
    """
    Validates that a numeric DataFrame column is within a specified range.
    """

    def __init__(
        self, 
        column_name: str, 
        logger: ValidatorLogger,
        min_value: Optional[float] = None, 
        max_value: Optional[float] = None, 
        inclusive: bool = True
    ) -> None:
        """
        :param column_name: The column to validate.
        :param min_value: Minimum allowed value (None for no lower bound).
        :param max_value: Maximum allowed value (None for no upper bound).
        :param logger: Logger instance.
        :param inclusive: If True, bounds are inclusive (<=, >=); otherwise, exclusive (<, >).
        """
        if min_value is None and max_value is None:
            raise ValueError("At least one of 'min_value' or 'max_value' must be specified.")
        
        super().__init__()
        self.column_name = column_name
        self.min_value = min_value
        self.max_value = max_value
        self.logger = logger
        self.inclusive = inclusive
        self.is_valid = True
        self._next_handler: Optional[BaseValidationHandler] = None

    def set_next(self, handler: BaseValidationHandler) -> BaseValidationHandler:
        """
        Set the next handler in the chain only if validation has passed.
        """
        if not self.is_valid:
            raise ValueError("Cannot set next handler because the current validation failed.")
        self._next_handler = handler
        return handler

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the column values fall within the specified numeric range.
        Logs errors but does not drop rows. Returns validation results.
        """
        errors = []

        if self.column_name not in df.columns:
            error_message = f"Column '{self.column_name}' not found in DataFrame."
            self.logger.log_failure(self.column_name, error_message)
            self.is_valid = False
            return {"success": False, "errors": [error_message]}

        # Convert column to numeric, coercing non-numeric values to NaN
        column_data = pd.to_numeric(df[self.column_name], errors="coerce")

        # Check for non-numeric values (NaNs after coercion)
        invalid_mask = column_data.isna()
        if invalid_mask.any():
            error_message = f"Column '{self.column_name}' contains {invalid_mask.sum()} non-numeric values."
            self.logger.log_failure(self.column_name, error_message)
            errors.append(error_message)

        # Validate range (only if column is numeric)
        if errors:
            self.is_valid = False
        else:
            if self.inclusive:
                out_of_range_mask = ((self.min_value is not None) & (column_data < self.min_value)) | \
                                    ((self.max_value is not None) & (column_data > self.max_value))
            else:
                out_of_range_mask = ((self.min_value is not None) & (column_data <= self.min_value)) | \
                                    ((self.max_value is not None) & (column_data >= self.max_value))

            if out_of_range_mask.any():
                error_message = f"Column '{self.column_name}' has {out_of_range_mask.sum()} out-of-range values."
                self.logger.log_failure(self.column_name, error_message)
                errors.append(error_message)
                self.is_valid = False
            else:
                self.logger.log_success(self.column_name)

        # Pass validation results to the next handler if valid
        if self.is_valid and self._next_handler:
            next_result = self._next_handler.validate(df)
            errors.extend(next_result["errors"])
            self.is_valid &= next_result["success"]

        return {"success": self.is_valid, "errors": errors}
