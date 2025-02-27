import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')

class StopwordRemover:
    """Removes stopwords from text columns in a DataFrame."""

    def __init__(self, language="english"):
        """
        Initializes the StopwordRemover with stopwords from the specified language.

        :param language: Language for stopword removal (default is English).
        """
        self.stopwords = set(stopwords.words(language))

    def process(self, df: pd.DataFrame, columns: list):
        """
        Removes stopwords from the specified text columns.

        :param df: Pandas DataFrame containing text data.
        :param columns: List of column names to process.
        :return: DataFrame with stopwords removed in specified columns.
        """

        def remove_stopwords(text):
            """Removes stopwords from a given text."""
            if isinstance(text, str):  # Ensure it's a string
                words = text.split()
                return " ".join(word for word in words if word.lower() not in self.stopwords)
            return text  # Return unchanged if not a string (e.g., NaN or non-text values)

        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(remove_stopwords)

        return df
