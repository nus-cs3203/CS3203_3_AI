import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('punkt')

class Tokenizer:
    def process(self, df: pd.DataFrame, columns: list):
        """
        Tokenizes text in the specified columns of the DataFrame.

        :param df: Pandas DataFrame containing text data.
        :param columns: List of column names to tokenize.
        :return: DataFrame with tokenized text in specified columns.
        """
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(word_tokenize)
        return df
