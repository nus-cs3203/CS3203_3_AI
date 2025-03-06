import pandas as pd
import logging

class DuplicateRemover:
    """Removes duplicate rows in a Pandas DataFrame."""

    def __init__(self, subset: list = None):
        """
        Initialize the duplicate remover.

        :param subset: List of column names to check for duplicates. If None, checks all columns.
        """
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)
        self.subset = subset

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows based on all or specific columns.

        :param df: Pandas DataFrame.
        :return: DataFrame without duplicate rows.
        """
        try:
            if df.empty:
                self.logger.warning("DataFrame is empty. No duplicates to remove.")
                return df

            before_count = len(df)
            df = df.drop_duplicates(subset=self.subset, keep="first").reset_index(drop=True)
            after_count = len(df)

            self.logger.info(f"Removed {before_count - after_count} duplicate rows.")
            return df

        except KeyError as e:
            self.logger.error(f"KeyError: {e}. Check if the subset columns exist in the DataFrame.")
            raise

        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise
