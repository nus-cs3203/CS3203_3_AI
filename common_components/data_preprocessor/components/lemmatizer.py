import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import logging

# Ensure necessary NLTK data is available
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")

class Lemmatizer:
    """Lemmatizes words in specified text columns of a Pandas DataFrame."""

    def __init__(self, text_columns):
        """
        :param text_columns: List of text columns to be lemmatized.
        """
        self.text_columns = text_columns
        self.lemmatizer = WordNetLemmatizer()
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies lemmatization to the specified text columns.

        :param df: Pandas DataFrame with text columns.
        :return: DataFrame with lemmatized text columns.
        """
        if df.empty:
            self.logger.warning("DataFrame is empty. No lemmatization applied.")
            return df

        for col in self.text_columns:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(str).apply(self._lemmatize_text)
                    self.logger.info(f"Lemmatization applied to column: {col}")
                except Exception as e:
                    self.logger.error(f"Error processing column '{col}': {e}")
            else:
                self.logger.warning(f"Column '{col}' not found in DataFrame.")

        return df

    def _lemmatize_text(self, text):
        """Helper function to lemmatize a single text entry with better tokenization."""
        if not isinstance(text, str):
            self.logger.warning(f"Expected string type but got {type(text)}. Returning original value.")
            return text  # Return as-is if not a string

        try:
            tokens = word_tokenize(text)  # More robust tokenization
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            return " ".join(lemmatized_tokens)
        except Exception as e:
            self.logger.error(f"Error lemmatizing text '{text}': {e}")
            return text  # Return original text if an error occurs
