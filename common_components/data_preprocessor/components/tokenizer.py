import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('punkt')

class Tokenizer:
    """Tokenizes text in specified columns of a DataFrame."""

    def __init__(self, text_columns):
        """
        :param text_columns: List of text columns to process.
        """
        self.text_columns = text_columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenizes text in the specified columns of the DataFrame.

        :param df: Pandas DataFrame containing text data.
        :return: DataFrame with tokenized text in specified columns.
        """
        def tokenize_text(text):
            """Tokenizes text if it is a valid string."""
            return word_tokenize(text) if isinstance(text, str) else text

        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].map(tokenize_text)

        return df
