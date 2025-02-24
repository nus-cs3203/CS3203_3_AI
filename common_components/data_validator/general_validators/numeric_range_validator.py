import numpy as np
import pandas as pd
from common_components.data_validator.validation_handler import ValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class NumericRangeValidator(ValidationHandler):
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
        :param inclusive: If True, uses <= and >= for bounds; otherwise, uses < and >.
        """
        super().__init__()
        if min_value is None and max_value is None:
            raise ValueError("At least one of 'min_value' or 'max_value' must be specified.")
        
        self.column_name = column_name
        self.min_value = min_value
        self.max_value = max_value
        self.logger = logger
        self.inclusive = inclusive

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the column values fall within the specified numeric range.
        """
        self.logger.log_dataframe(df)

        if self.column_name not in df.columns:
            error_message = f"Validation failed: Column '{self.column_name}' not found in DataFrame."
            self.logger.log_failure(self.column_name, error_message)
            return {"error": error_message}

        # Ensure the column is numeric
        if not np.issubdtype(df[self.column_name].dtype, np.number):
            error_message = f"Validation failed: '{self.column_name}' must be a numeric column."
            self.logger.log_failure(self.column_name, error_message)
            return {"error": error_message}

        # Identify invalid rows
        if self.inclusive:
            invalid_rows = df[
                (df[self.column_name] < self.min_value) if self.min_value is not None else False |
                (df[self.column_name] > self.max_value) if self.max_value is not None else False
            ]
        else:
            invalid_rows = df[
                (df[self.column_name] <= self.min_value) if self.min_value is not None else False |
                (df[self.column_name] >= self.max_value) if self.max_value is not None else False
            ]

        if not invalid_rows.empty:
            error_message = f"Validation failed: '{self.column_name}' has out-of-range values in rows {list(invalid_rows.index)}."
            self.logger.log_failure(self.column_name, error_message)
            return {"error": error_message, "invalid_rows": invalid_rows.to_dict(orient='records')}

        self.logger.log_success(self.column_name)
        return self._validate_next(df)
