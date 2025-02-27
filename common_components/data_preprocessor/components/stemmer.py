import pandas as pd
from nltk.stem import PorterStemmer

class Stemmer:
    """Applies stemming to text columns in a DataFrame."""

    def __init__(self):
        self.stemmer = PorterStemmer()

    def process(self, df: pd.DataFrame, columns: list):
        """
        Applies stemming to specified text columns.

        :param df: Pandas DataFrame containing text data.
        :param columns: List of column names to process.
        :return: DataFrame with stemmed words in specified columns.
        """

        def stem_text(text):
            """Stems each word in the given text."""
            if isinstance(text, str):  # Ensure it's a string
                words = text.split()
                return " ".join(self.stemmer.stem(word) for word in words)
            return text  # Return unchanged if not a string (e.g., NaN or non-text values)

        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(stem_text)

        return df
