import pandas as pd
import logging

class ColumnsJoiner:
    """Joins multiple columns into a single column."""

    def __init__(self, subset: list = None, join_column_name: str = "title_with_desc"):
        """
        Initialize the columns joiner.

        :param subset: List of column names to join. If None, joins no columns.
        :param join_column_name: Name of the new column to create by joining the subset columns.
        """
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)
        self.subset = subset
        self.join_column_name = join_column_name

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Joins specific columns.

        :param df: Pandas DataFrame.
        :return: DataFrame with new column.
        """
        if df.empty:
            self.logger.warning("DataFrame is empty. No columns to join.")
            return df

        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input is not a pandas DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        if self.subset:
            missing_columns = [col for col in self.subset if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing columns in DataFrame: {missing_columns}")
                raise ValueError(f"DataFrame does not contain columns: {missing_columns}")

            try:
                self.logger.info(f"Joining columns {self.subset} into {self.join_column_name}")
                df[self.join_column_name] = df[self.subset].astype(str).agg(' '.join, axis=1)
                self.logger.info(f"Successfully joined columns into {self.join_column_name}")
            except Exception as e:
                self.logger.error(f"Error while joining columns: {e}")
                raise

        return df
