import pandas as pd
import re

class Normalizer:
    """Normalizes text columns in a Pandas DataFrame."""

    def process(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Applies normalization to specified text columns.

        :param df: Pandas DataFrame containing text data.
        :param columns: List of column names to normalize.
        :return: DataFrame with normalized text.
        """

        def normalize_text(text):
            """Applies normalization rules to the given text."""
            if not isinstance(text, str):
                return text  # Return unchanged if not a string (e.g., NaN or non-text values)
            
            text = text.lower().strip()
            text = re.sub(r'[:;=xX8][-~]?[)DPO]', '<smiley>', text)  # Replace happy emoticons
            text = re.sub(r'[:;=][-~]?[\(\[]', '<sad>', text)  # Replace sad emoticons
            return text

        valid_columns = [col for col in columns if col in df.columns]

        if not valid_columns:
            print("Warning: No valid text columns found for normalization. Skipping process.")
            return df  # Return unchanged if no valid columns exist

        df = df.copy()  # Avoid modifying the original DataFrame
        for col in valid_columns:
            df[col] = df[col].astype(str).apply(normalize_text)  # Ensure text type before processing

        return df
