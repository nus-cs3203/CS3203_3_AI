import pandas as pd
import re

class Normalizer:
    """Normalizes text columns in a Pandas DataFrame."""

    def process(self, df: pd.DataFrame, columns: list):
        """
        Applies normalization to specified text columns.

        :param df: Pandas DataFrame containing text data.
        :param columns: List of column names to normalize.
        :return: DataFrame with normalized text.
        """

        def normalize_text(text):
            """Applies normalization rules to the given text."""
            if isinstance(text, str):  # Ensure it's a string
                text = text.lower().strip()
                text = re.sub(r'[:;xX8=][-~]?[)DPO]', '<smiley>', text)  # Replace common emoticons
                text = re.sub(r'[:;][-~]?(\()', '<sad>', text)  # Replace sad emoticons
                return text
            return text  # Return unchanged if not a string (e.g., NaN or non-text values)

        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(normalize_text)

        return df
