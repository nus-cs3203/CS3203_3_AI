import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import logging

nltk.download("stopwords")
nltk.download("punkt")

class StopwordRemover:
    """Removes stopwords from specified text columns in a DataFrame."""

    def __init__(self, text_columns, language="english"):
        """
        Initializes the StopwordRemover with stopwords from the specified language.

        :param text_columns: List of column names to process.
        :param language: Language for stopword removal (default is English).
        """
        self.text_columns = text_columns
        logging.basicConfig(level=logging.WARNING)        
        self.logger = logging.getLogger(__name__)
        try:
            self.stopwords = set(stopwords.words(language))
            self.logger.info(f"Stopwords for language '{language}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading stopwords for language '{language}': {e}")
            self.stopwords = set()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes stopwords from the specified text columns.

        :param df: Pandas DataFrame containing text data.
        :return: DataFrame with stopwords removed in specified columns.
        """
        for col in self.text_columns:
            if (col in df.columns):
                try:
                    df[col] = df[col].apply(self._remove_stopwords)
                    self.logger.info(f"Processed column '{col}' successfully.")
                except Exception as e:
                    self.logger.error(f"Error processing column '{col}': {e}")
            else:
                self.logger.warning(f"Column '{col}' not found in DataFrame.")
        return df

    def _remove_stopwords(self, text):
        """Removes stopwords from a single text entry."""
        if isinstance(text, str):
            try:
                words = word_tokenize(text)  # Tokenization ensures better handling of punctuation
                return " ".join(word for word in words if word.lower() not in self.stopwords)
            except Exception as e:
                self.logger.error(f"Error removing stopwords from text: {e}")
                return text
        return text  # Return unchanged if not a string
