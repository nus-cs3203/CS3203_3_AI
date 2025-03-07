import numpy as np
import pandas as pd
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger
from typing import Optional, List

class NumericRangeValidator(BaseValidationHandler):
    """
    Validates that numeric DataFrame columns are within a specified range.
    """

    def __init__(
        self, 
        column_names: List[str], 
        logger: ValidatorLogger,
        min_value: Optional[float] = None, 
        max_value: Optional[float] = None, 
        inclusive: bool = True
    ) -> None:
        """
        :param column_names: The columns to validate.
        :param min_value: Minimum allowed value (None for no lower bound).
        :param max_value: Maximum allowed value (None for no upper bound).
        :param logger: Logger instance.
        :param inclusive: If True, bounds are inclusive (<=, >=); otherwise, exclusive (<, >).
        """
        if min_value is None and max_value is None:
            raise ValueError("At least one of 'min_value' or 'max_value' must be specified.")
        
        super().__init__()
        self.column_names = column_names
        self.min_value = min_value
        self.max_value = max_value
        self.logger = logger
        self.inclusive = inclusive
        self._next_handler: Optional[BaseValidationHandler] = None

    def set_next(self, handler: BaseValidationHandler) -> BaseValidationHandler:
        """
        Set the next handler in the chain only if validation has passed.
        """
        self._next_handler = handler
        return handler

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the column values fall within the specified numeric range.
        Logs errors but does not drop rows. Returns validation results.
        """
        self.logger.log_dataframe(df)
        errors = []

        for column_name in self.column_names:
            if column_name not in df.columns:
                error_message = f"Column '{column_name}' not found in DataFrame."
                self.logger.log_failure(column_name, error_message)
                raise ValueError("Current validation failed due to missing column(s).")

            # Replace nan (not responsibility of this validator) with min_value 
            column_data = df[column_name].replace(np.nan, self.min_value)

            # Validate range (only if column is numeric)
            if self.inclusive:
                out_of_range_mask = ((self.min_value is not None) & (column_data < self.min_value)) | \
                                    ((self.max_value is not None) & (column_data > self.max_value))
            else:
                out_of_range_mask = ((self.min_value is not None) & (column_data <= self.min_value)) | \
                                    ((self.max_value is not None) & (column_data >= self.max_value))

            if out_of_range_mask.any():
                error_message = f"Column '{column_name}' has {out_of_range_mask.sum()} out-of-range values."
                self.logger.log_failure(column_name, error_message)
                errors.append(error_message)
                raise ValueError(error_message)
            else:
                self.logger.log_success(column_name)

        # Pass validation to the next handler if valid
        return self._validate_next(df)

    def _validate_next(self, df: pd.DataFrame) -> dict:
        """
        Pass the DataFrame to the next handler in the chain if it exists.
        """
        if self._next_handler:
            return self._next_handler.validate(df)

        return {"success": True}
