import pandas as pd
import numpy as np
from common_components.data_validator.validation_handler import ValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class NotEmptyValidator(ValidationHandler):
    """
    Validates that a given DataFrame column is not None, empty, or NaN.
    Drops rows that fail validation.
    """

    def __init__(self, column_names: list, logger: ValidatorLogger = None) -> None:
        super().__init__()
        self.column_names = column_names
        self.logger = logger

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that the specified columns do not contain null, empty, or NaN values.
        Invalid rows are dropped from the DataFrame.
        """
        if self.logger:
            self.logger.log_dataframe(df)

        valid_df = df.copy()
        for column in self.column_names:
            if column not in valid_df.columns:
                error_message = f"Validation failed: Column '{column}' not found in DataFrame."
                if self.logger:
                    self.logger.log_failure(column, error_message)
                continue  # Skip validation for missing columns

            # Drop rows where the column is empty or NaN
            invalid_rows = valid_df[
                valid_df[column].isna() | 
                (valid_df[column].astype(str).str.strip() == "")
            ]

            if not invalid_rows.empty:
                error_message = f"Validation failed: '{column}' contains empty/null values. Dropping {len(invalid_rows)} rows."
                if self.logger:
                    self.logger.log_failure(column, error_message)
                
                valid_df = valid_df.drop(invalid_rows.index)  # Drop invalid rows

            else:
                if self.logger:
                    self.logger.log_success(column)

        # Return the cleaned DataFrame
        return self._validate_next(valid_df)
