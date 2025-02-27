import pandas as pd

class DuplicateRemover:
    """Removes duplicate rows in a Pandas DataFrame."""

    def __init__(self, subset=None):
        """
        :param subset: List of column names to check for duplicates. If None, checks all columns.
        """
        self.subset = subset  # Subset of columns for checking duplicates

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows based on all or specific columns.

        :param df: Pandas DataFrame.
        :return: DataFrame without duplicate rows.
        """
        return df.drop_duplicates(subset=self.subset, keep="first").reset_index(drop=True)
