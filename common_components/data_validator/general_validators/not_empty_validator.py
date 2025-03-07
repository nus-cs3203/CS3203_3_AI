import pandas as pd
import numpy as np
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger
from typing import Optional, List

class NotEmptyValidator(BaseValidationHandler):
    """
    Validates that specified DataFrame columns are not None, empty, or NaN.
    """

    def __init__(self, column_names: List[str], logger: Optional[ValidatorLogger] = None) -> None:
        """
        :param column_names: List of columns to check for emptiness or NaN values.
        :param logger: Optional logger instance. Uses default logger if not provided.
        """
        super().__init__()
        self.column_names = column_names
        self.logger = logger or ValidatorLogger()
        self._next_handler: Optional[BaseValidationHandler] = None  # Only set if validation passes

    def set_next(self, handler: BaseValidationHandler) -> BaseValidationHandler:
        """
        Set the next handler in the chain only if the current validation has passed.
        """
        self._next_handler = handler
        return handler

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the specified columns do not contain null, empty, or NaN values.
        """
        self.logger.log_dataframe(df)

        missing_cols = [col for col in self.column_names if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                self.logger.log_failure(col, f"Validation failed: Column '{col}' not found in DataFrame.")
            raise ValueError("Current validation failed due to missing columns.")
        
        # Identify invalid rows
        invalid_mask = df[self.column_names].isna() | (df[self.column_names].applymap(str).applymap(str.strip) == "")
        invalid_rows = df[invalid_mask.any(axis=1)]

        if not invalid_rows.empty:
            self.is_valid = False
            error_msg = f"Validation failed: {len(invalid_rows)} rows contain empty/null values."
            self.logger.log_failure(", ".join(self.column_names), error_msg)
            raise ValueError("Current validation failed due to empty/null values.")

        # Log success for each validated column
        for col in self.column_names:
            self.logger.log_success(col)

        # Pass to next handler if it exists
        return self._validate_next(df)

    def _validate_next(self, df: pd.DataFrame) -> dict:
        """
        Pass the DataFrame to the next handler in the chain if it exists.
        """
        if self._next_handler:
            return self._next_handler.validate(df)

        return {"success": True}
