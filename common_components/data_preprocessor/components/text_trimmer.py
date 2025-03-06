import re
import pandas as pd
import logging

class TextTrimmer:
    """Cleans up extra spaces in text fields within specified columns."""

    def __init__(self, text_columns):
        """
        :param text_columns: List of text columns to process.
        """
        self.text_columns = text_columns
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trims extra spaces in the specified text columns of the DataFrame.

        :param df: Pandas DataFrame containing text data.
        :return: DataFrame with trimmed text in specified columns.
        """
        def trim_text(text):
            """Trims leading/trailing spaces and collapses multiple spaces within the text."""
            if isinstance(text, str):
                return re.sub(r"\s{2,}", " ", text).strip()
            return text  # Return non-string values unchanged

        for col in self.text_columns:
            if col in df.columns:
                self.logger.info(f"Processing column: {col}")
                try:
                    df[col] = df[col].map(trim_text)  # More efficient than apply
                except Exception as e:
                    self.logger.error(f"Error processing column {col}: {e}")
            else:
                self.logger.warning(f"Column {col} does not exist in the DataFrame")

        return df
