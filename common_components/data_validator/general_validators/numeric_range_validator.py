import numpy as np
import pandas as pd
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class NumericRangeValidator(BaseValidationHandler):
    """
    Validates that a numeric DataFrame column is within a specified range.
    """
    
    def __init__(
        self, 
        column_name: str, 
        logger: ValidatorLogger,
        min_value: float = None, 
        max_value: float = None, 
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
        
        self.column_name = column_name
        self.min_value = min_value
        self.max_value = max_value
        self.logger = logger
        self.inclusive = inclusive
        self.next_handler = None  # Next handler in chain

    def set_next(self, handler: 'BaseValidationHandler') -> 'BaseValidationHandler':
        self.next_handler = handler
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
            errors.append(error_message)
        else:
            df[self.column_name] = pd.to_numeric(df[self.column_name], errors="coerce")
            invalid_mask = df[self.column_name].isna()
            if invalid_mask.any():
                error_message = f"Column '{self.column_name}' contains non-numeric values."
                self.logger.log_failure(self.column_name, error_message)
                errors.append(error_message)
            
            if self.inclusive:
                out_of_range_mask = ((self.min_value is not None) & (df[self.column_name] < self.min_value)) | \
                                    ((self.max_value is not None) & (df[self.column_name] > self.max_value))
            else:
                out_of_range_mask = ((self.min_value is not None) & (df[self.column_name] <= self.min_value)) | \
                                    ((self.max_value is not None) & (df[self.column_name] >= self.max_value))
            
            if out_of_range_mask.any():
                error_message = f"Column '{self.column_name}' has {out_of_range_mask.sum()} out-of-range values."
                self.logger.log_failure(self.column_name, error_message)
                errors.append(error_message)
            else:
                self.logger.log_success(self.column_name)
        
        # Pass validation results to next handler
        result = {"success": not errors, "errors": errors}
        if self.next_handler:
            next_result = self.next_handler.validate(df)
            result["errors"].extend(next_result["errors"])
            result["success"] &= next_result["success"]
        
        return result