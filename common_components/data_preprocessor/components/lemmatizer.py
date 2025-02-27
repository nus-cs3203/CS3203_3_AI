import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

class Lemmatizer:
    """Lemmatizes words in specified text columns of a Pandas DataFrame."""

    def __init__(self, text_columns):
        """
        :param text_columns: List of text columns to be lemmatized.
        """
        self.text_columns = text_columns
        self.lemmatizer = WordNetLemmatizer()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies lemmatization to the specified text columns.

        :param df: Pandas DataFrame with text columns.
        :return: DataFrame with lemmatized text columns.
        """
        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._lemmatize_text)
        return df

    def _lemmatize_text(self, text):
        """Helper function to lemmatize a single text entry."""
        if isinstance(text, str):
            tokens = text.split()  # Tokenize based on spaces
            return " ".join([self.lemmatizer.lemmatize(token) for token in tokens])
        return text  # Return as-is if not a string
