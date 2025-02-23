import re
import pandas as pd

class TextTrimmer:
    """Cleans up extra spaces in text fields within specified columns."""

    def process(self, df: pd.DataFrame, columns: list):
        """
        Trims extra spaces in the specified text columns of the DataFrame.

        :param df: Pandas DataFrame containing text data.
        :param columns: List of column names to trim.
        :return: DataFrame with trimmed text in specified columns.
        """
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: re.sub(r"\s+", " ", x).strip())
        return df
