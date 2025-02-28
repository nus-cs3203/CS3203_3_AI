import pandas as pd
import numpy as np
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validation_handler import ValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class NotEmptyValidator(BaseValidationHandler):
    """
    Validates that specified DataFrame columns are not None, empty, or NaN.
    Drops rows that fail validation.
    """

    def __init__(self, column_names: list, logger: ValidatorLogger = None) -> None:
        """
        :param column_names: List of columns to check for emptiness or NaN values.
        :param logger: Optional logger instance. Uses default logger if not provided.
        """
        super().__init__()
        self.column_names = column_names
        self.logger = logger or ValidatorLogger()  # Use a default logger if none is provided

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates that the specified columns do not contain null, empty, or NaN values.
        Invalid rows are dropped from the DataFrame.
        """
        self.logger.log_dataframe(df)

        missing_cols = [col for col in self.column_names if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                self.logger.log_failure(col, f"Validation failed: Column '{col}' not found in DataFrame. Skipping validation.")
            return self._validate_next(df)  # Skip processing if all columns are missing

        # Identify invalid rows
        invalid_mask = df[self.column_names].isna() | (df[self.column_names].astype(str).str.strip() == "")
        invalid_rows = df[invalid_mask.any(axis=1)]

        if not invalid_rows.empty:
            self.logger.log_failure(
                ", ".join(self.column_names),
                f"Validation failed: {len(invalid_rows)} rows contain empty/null values. Dropping them."
            )
            df = df.drop(index=invalid_rows.index).reset_index(drop=True)
        else:
            for col in self.column_names:
                self.logger.log_success(col)

        return self._validate_next(df)  # Continue to next handler in the chain
