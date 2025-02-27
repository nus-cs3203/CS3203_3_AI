import pandas as pd

class MissingValueHandler:
    """Removes rows that have missing values in critical columns."""

    def __init__(self, critical_columns):
        """
        :param critical_columns: List of columns that must not have missing values.
        """
        self.critical_columns = critical_columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows that contain missing (NaN) values in critical columns.

        :param df: Pandas DataFrame containing data.
        :return: DataFrame with rows having missing critical values removed.
        """
        return df.dropna(subset=self.critical_columns)
