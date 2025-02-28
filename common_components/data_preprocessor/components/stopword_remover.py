import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

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
        self.stopwords = set(stopwords.words(language))

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes stopwords from the specified text columns.

        :param df: Pandas DataFrame containing text data.
        :return: DataFrame with stopwords removed in specified columns.
        """
        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._remove_stopwords)
        return df

    def _remove_stopwords(self, text):
        """Removes stopwords from a single text entry."""
        if isinstance(text, str):
            words = word_tokenize(text)  # Tokenization ensures better handling of punctuation
            return " ".join(word for word in words if word.lower() not in self.stopwords)
        return text  # Return unchanged if not a string
