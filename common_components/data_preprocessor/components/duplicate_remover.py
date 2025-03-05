import pandas as pd
import logging

class DuplicateRemover:
    """Removes duplicate rows in a Pandas DataFrame."""

    def __init__(self, subset: list = None):
        """
        Initialize the duplicate remover.

        :param subset: List of column names to check for duplicates. If None, checks all columns.
        """

        self.logger = logging.getLogger(__name__)
        self.subset = subset

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df = df.drop_duplicates(subset=self.subset, keep="first").reset_index(drop=True)
        after_count = len(df)

        if self.logger:
            self.logger.info(f"Removed {before_count - after_count} duplicate rows.")

        return df
