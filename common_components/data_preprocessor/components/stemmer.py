import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import logging

nltk.download("punkt")

class Stemmer:
    """Applies stemming to text columns in a DataFrame."""

    def __init__(self, text_columns):
        """
        :param text_columns: List of text columns to process.
        """
        self.text_columns = text_columns
        self.stemmer = PorterStemmer()
        logging.basicConfig(level=logging.WARNING)        
        self.logger = logging.getLogger(__name__)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies stemming to the specified text columns.

        :param df: Pandas DataFrame containing text data.
        :return: DataFrame with stemmed words in specified columns.
        """
        for col in self.text_columns:
            if (col in df.columns):
                try:
                    df[col] = df[col].apply(self._stem_text)
                except Exception as e:
                    self.logger.error(f"Error processing column '{col}': {e}")
            else:
                self.logger.warning(f"Column '{col}' not found in DataFrame.")
        return df

    def _stem_text(self, text):
        """Helper function to stem each word in a text entry."""
        if isinstance(text, str):
            try:
                words = word_tokenize(text)  # More robust tokenization
                return " ".join(self.stemmer.stem(word) for word in words)
            except Exception as e:
                self.logger.error(f"Error stemming text '{text}': {e}")
                return text
        return text