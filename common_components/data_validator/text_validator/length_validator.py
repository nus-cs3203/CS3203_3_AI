import pandas as pd
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger
from typing import Dict, Tuple, Optional

class LengthValidator(BaseValidationHandler):
    """
    Validates that specified text columns have string lengths within the given range.
    """

    def __init__(self, text_cols: Dict[str, Tuple[int, int]], logger: Optional[ValidatorLogger] = None) -> None:
        """
        :param text_cols: Dictionary where keys are column names, and values are (min_length, max_length) tuples.
        :param logger: Optional logger instance. Uses default logger if not provided.
        """
        super().__init__()
        self.text_cols = text_cols
        self.logger = logger or ValidatorLogger()
        self._next_handler: Optional[BaseValidationHandler] = None

    def set_next(self, handler: BaseValidationHandler) -> BaseValidationHandler:
        """
        Set the next handler in the chain only if the current validation has passed.
        """
        self._next_handler = handler
        return handler

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates the length of text columns.
        :param df: pandas DataFrame.
        :return: Dictionary with success status and error details if validation fails.
        """
        self.logger.log_dataframe(df)

        missing_cols = [col for col in self.text_cols.keys() if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                self.logger.log_failure(col, f"Validation failed: Column '{col}' not found in DataFrame.")
            raise ValueError("Current validation failed due to missing columns.")

        errors = []
        for col, (min_length, max_length) in self.text_cols.items():
            valid_mask = df[col].notna()
            valid_texts = df.loc[valid_mask, col].astype(str).str.strip()

            # Ensure the dtype is string before measuring length
            text_lengths = valid_texts[valid_texts != ""].str.len()

            # Identify invalid rows
            invalid_rows = text_lengths[~text_lengths.between(min_length, max_length)].index.tolist()

            if invalid_rows:
                error_message = f"Column '{col}' must have length between {min_length} and {max_length}. Found {len(invalid_rows)} invalid rows."
                self.logger.log_failure(col, error_message)
                errors.append(error_message)
                self.is_valid = False
            else:
                self.logger.log_success(col)

        if errors:
            raise ValueError(f"Current validation failed due to invalid rows: {errors}")

        # Pass to next handler if it exists
        return self._validate_next(df)

    def _validate_next(self, df: pd.DataFrame) -> dict:
        """
        Pass the DataFrame to the next handler in the chain if it exists.
        """
        if self._next_handler:
            return self._next_handler.validate(df)

        return {"success": True}
