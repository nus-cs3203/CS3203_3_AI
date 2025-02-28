import pandas as pd
import logging

class DuplicateRemover:
    """Removes duplicate rows in a Pandas DataFrame."""

    def __init__(self):
        """Initialize the duplicate remover."""
        self.logger = logging.getLogger(__name__)

    def process(self, df: pd.DataFrame, subset=None) -> pd.DataFrame:
        """
        Removes duplicate rows based on all or specific columns.

        :param df: Pandas DataFrame.
        :param subset: List of column names to check for duplicates. If None, checks all columns.
        :return: DataFrame without duplicate rows.
        """
        if df.empty:
            if self.logger:
                self.logger.warning("DataFrame is empty. No duplicates to remove.")
            return df

        before_count = len(df)
        df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
        after_count = len(df)

        if self.logger:
            self.logger.info(f"Removed {before_count - after_count} duplicate rows.")

        return df
