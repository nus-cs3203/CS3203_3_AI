import pandas as pd
import re
import logging

class Normalizer:
    """Normalizes text columns in a Pandas DataFrame."""

    def __init__(self, text_columns: list):
        """
        Initializes the Normalizer with the specified text columns.

        :param text_columns: List of column names containing text data.
        """
        self.text_columns = text_columns
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies normalization to specified text columns.

        :param df: Pandas DataFrame containing text data.
        :return: DataFrame with normalized text.
        """

        def normalize_text(text):
            """Applies normalization rules to the given text."""
            if not isinstance(text, str):
                return text  # Return unchanged if not a string (e.g., NaN or non-text values)
            
            if pd.isna(text) or text.lower() in ["none", "[deleted]"]:
                return "[Unknown]"
            text = text.lower().strip()
            text = re.sub(r'[:;=xX8][-~]?[)DPO]', '<smiley>', text)  # Replace happy emoticons
            text = re.sub(r'[:;=][-~]?[\(\[]', '<sad>', text)  # Replace sad emoticons
            return text

        try:
            valid_columns = [col for col in self.text_columns if col in df.columns]

            if not valid_columns:
                self.logger.warning("No valid text columns found for normalization. Skipping process.")
                return df  # Return unchanged if no valid columns exist

            for col in valid_columns:
                df[col] = df[col].astype(str).apply(normalize_text)  # Ensure text type before processing

            self.logger.info("Normalization process completed successfully.")
            return df

        except Exception as e:
            self.logger.error(f"An error occurred during normalization: {e}")
            raise
