import pandas as pd
import numpy as np
from common_components.data_validator.validation_handler import ValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class NotEmptyValidator(ValidationHandler):
    """
    Validates that a given DataFrame column is not None, empty, or NaN.
    """

    def __init__(self, column_name: str, logger: ValidatorLogger) -> None:
        super().__init__()
        self.column_name = column_name
        self.logger = logger

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validate that the column does not contain null, empty, or NaN values.
        """
        self.logger.log_dataframe(df)

        if self.column_name not in df.columns:
            error_message = f"Validation failed: Column '{self.column_name}' not found in DataFrame."
            self.logger.log_failure(self.column_name, error_message)
            return {"error": error_message}

        # Identify invalid rows
        invalid_rows = df[
            df[self.column_name].isna() | 
            (df[self.column_name].astype(str).str.strip() == "")  # Empty strings
        ]

        if not invalid_rows.empty:
            error_message = f"Validation failed: '{self.column_name}' contains empty/null values in rows {list(invalid_rows.index)}."
            self.logger.log_failure(self.column_name, error_message)
            return {"error": error_message, "invalid_rows": invalid_rows.to_dict(orient="records")}

        self.logger.log_success(self.column_name)
        return self._validate_next(df)
