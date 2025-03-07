import re
import pandas as pd
from common_components.data_validator.base_handler import BaseValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger
from typing import List, Optional

class RegexValidator(BaseValidationHandler):
    """
    Validates that specified string fields in a DataFrame match a given regex pattern.
    """

    def __init__(self, columns: List[str], patterns: List[str], logger: ValidatorLogger, allow_nan: bool = True, log_sample: int = 3) -> None:
        """
        :param columns: List of column names to validate.
        :param patterns: List of corresponding regex patterns (must be same length as `columns`).
        :param logger: Logger instance.
        :param allow_nan: Whether NaN values should be treated as valid (default: True).
        :param log_sample: Number of sample invalid values to log for debugging.
        """
        super().__init__()

        if len(columns) != len(patterns):
            raise ValueError("`columns` and `patterns` must have the same length.")

        self.columns = columns
        self.patterns = [re.compile(p) for p in patterns]  # Precompile regex for efficiency
        self.logger = logger
        self.allow_nan = allow_nan
        self.log_sample = log_sample
        self._next_handler: Optional[BaseValidationHandler] = None

    def set_next(self, handler: BaseValidationHandler) -> BaseValidationHandler:
        """
        Sets the next handler in the chain, only if validation has passed.
        """
        self._next_handler = handler
        return handler

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validates that the specified columns in the DataFrame match their respective regex patterns.
        :param df: pandas DataFrame.
        :return: Dictionary with success status and error details.
        """
        self.logger.log_dataframe(df)
        errors = []

        # Iterate through each column and its corresponding pattern
        for col, regex in zip(self.columns, self.patterns):
            if col not in df.columns:
                warning_message = f"Warning: Column '{col}' not found in DataFrame. Skipping validation."
                self.logger.log_warning(col, warning_message)
                continue

            # Ensure NaN values are allowed if specified
            valid_mask = df[col].notna()  # Identify non-null values
            str_values = df.loc[valid_mask, col].astype(str)  # Convert only non-null values to string
            matched_mask = str_values.str.fullmatch(regex)

            # Combine with NaN handling
            final_mask = matched_mask | (~valid_mask if self.allow_nan else False)
            invalid_rows = df.index[~final_mask].tolist()

            if invalid_rows:
                sample_invalid_values = df.loc[invalid_rows, col].head(self.log_sample).tolist()
                error_message = (
                    f"Column '{col}' contains invalid values. "
                    f"Examples: {sample_invalid_values} (Found {len(invalid_rows)} invalid rows)"
                )
                self.logger.log_failure(col, error_message)
                errors.append(error_message)
            else:
                self.logger.log_success(col)

        validation_result = {"success": not errors, "errors": errors}

        # Raise error if validation fails
        if errors:
            raise ValueError(f"Validation failed. Errors: {errors}")

        # Pass to next handler if exists and validation passed
        if self._next_handler:
            next_result = self._validate_next(df)
            validation_result["errors"].extend(next_result.get("errors", []))
            validation_result["success"] = validation_result["success"] and next_result["success"]

        return validation_result

    def _validate_next(self, df: pd.DataFrame) -> dict:
        """
        Pass the DataFrame to the next handler in the chain if it exists.
        """
        if self._next_handler:
            return self._next_handler.validate(df)

        return {"success": True}
