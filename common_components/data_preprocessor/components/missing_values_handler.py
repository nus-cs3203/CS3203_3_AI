import pandas as pd
import logging

class MissingValueHandler:
    """Removes rows that have missing values in critical columns."""

    def __init__(self, critical_columns=None):
        """
        :param critical_columns: List of columns that must not have missing values.
        """
        self.critical_columns = critical_columns or []  # Ensure it's a list
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows that contain missing (NaN) values in critical columns.

        :param df: Pandas DataFrame containing data.
        :return: DataFrame with rows having missing critical values removed.
        """
        if df is None:
            self.logger.error("Input DataFrame is None.")
            raise ValueError("Input DataFrame cannot be None.")

        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input is not a pandas DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        if df.empty:
            self.logger.warning("DataFrame is empty. No missing value handling applied.")
            return df

        if not self.critical_columns:
            self.logger.warning("No critical columns specified. Skipping missing value handling.")
            return df

        valid_columns = [col for col in self.critical_columns if col in df.columns]

        if not valid_columns:
            self.logger.warning("None of the specified critical columns exist in DataFrame.")
            return df

        before_rows = len(df)
        df = df.dropna(subset=valid_columns)
        after_rows = len(df)

        removed_rows = before_rows - after_rows
        if removed_rows > 0:
            self.logger.info(f"Removed {removed_rows} rows with missing values in {valid_columns}")
        else:
            self.logger.info("No missing values found in the specified critical columns.")

        return df
