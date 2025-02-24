import logging
import pandas as pd

class ValidatorLogger:
    """
    A logger component for logging validation attempts and results in a pandas DataFrame.
    """

    def __init__(self, log_level=logging.INFO):
        """
        Initialize the logger with a given log level.
        """
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger("ValidatorLogger")

    def log_success(self, column_name: str):
        """Log a successful validation for a column."""
        self.logger.info(f"Validation passed for column: {column_name}")

    def log_failure(self, column_name: str, error_message: str):
        """Log a failed validation for a column."""
        self.logger.warning(f"Validation failed for column '{column_name}': {error_message}")

    def log_dataframe(self, df: pd.DataFrame):
        """Log details of the incoming DataFrame."""
        self.logger.debug(f"Processing DataFrame with shape: {df.shape} and columns: {list(df.columns)}")
