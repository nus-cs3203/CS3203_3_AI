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
        super().__init__()
        if min_value is None and max_value is None:
            raise ValueError("At least one of 'min_value' or 'max_value' must be specified.")
        
        self.column_name = column_name
        self.min_value = min_value
        self.max_value = max_value
        self.logger = logger
        self.inclusive = inclusive

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates that the column values fall within the specified numeric range.
        Logs errors and drops invalid rows.
        """
        self.logger.log_dataframe(df)

        if self.column_name not in df.columns:
            error_message = f"Validation failed: Column '{self.column_name}' not found in DataFrame."
            self.logger.log_failure(self.column_name, error_message)
            return self._validate_next(df)  # Continue processing

        # Ensure the column is numeric (coerce non-numeric values to NaN)
        df[self.column_name] = pd.to_numeric(df[self.column_name], errors="coerce")

        # Drop NaN values caused by non-numeric data
        invalid_rows = df[df[self.column_name].isna()]
        if not invalid_rows.empty:
            self.logger.log_failure(self.column_name, f"Dropping {len(invalid_rows)} rows with non-numeric values.")
            df = df.drop(index=invalid_rows.index)

        # Identify out-of-range rows
        if self.inclusive:
            out_of_range_mask = ((self.min_value is not None) & (df[self.column_name] < self.min_value)) | \
                                ((self.max_value is not None) & (df[self.column_name] > self.max_value))
        else:
            out_of_range_mask = ((self.min_value is not None) & (df[self.column_name] <= self.min_value)) | \
                                ((self.max_value is not None) & (df[self.column_name] >= self.max_value))

        invalid_rows = df[out_of_range_mask]

        # Log and drop invalid rows
        if not invalid_rows.empty:
            self.logger.log_failure(
                self.column_name,
                f"Validation failed: '{self.column_name}' has out-of-range values. Dropping {len(invalid_rows)} rows."
            )
            df = df.drop(index=invalid_rows.index)

        else:
            self.logger.log_success(self.column_name)

        return self._validate_next(df.reset_index(drop=True))  # Ensure reset index
