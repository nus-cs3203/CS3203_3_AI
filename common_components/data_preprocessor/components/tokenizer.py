import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import logging

nltk.download('punkt')

class Tokenizer:
    """Tokenizes text in specified columns of a DataFrame."""

    def __init__(self, text_columns):
        """
        :param text_columns: List of text columns to process.
        """
        self.text_columns = text_columns
        logging.basicConfig(level=logging.WARNING)        
        self.logger = logging.getLogger(__name__)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenizes text in the specified columns of the DataFrame.

        :param df: Pandas DataFrame containing text data.
        :return: DataFrame with tokenized text in specified columns.
        """
        def tokenize_text(text):
            """Tokenizes text if it is a valid string."""
            try:
                return word_tokenize(text) if isinstance(text, str) else text
            except Exception as e:
                self.logger.error(f"Error tokenizing text: {text}. Error: {e}")
                return text

        for col in self.text_columns:
            if col in df.columns:
                self.logger.info(f"Tokenizing column: {col}")
                df[col] = df[col].map(tokenize_text)
            else:
                self.logger.warning(f"Column {col} not found in DataFrame")

        return df
