import numpy as np
import pandas as pd
from gensim.models import Word2Vec

class TextEncoder:
    """Encodes text columns using a Word2Vec model."""

    def __init__(self, model_path=None):
        """
        Initializes the TextEncoder with an optional pre-trained Word2Vec model.

        :param model_path: Path to a pre-trained Word2Vec model (optional).
        """
        if model_path:
            self.model = Word2Vec.load(model_path)  # Load a pre-trained model
        else:
            self.model = None

    def process(self, df: pd.DataFrame, columns: list):
        """
        Encodes text in the specified columns using Word2Vec embeddings.

        :param df: Pandas DataFrame containing text data.
        :param columns: List of column names to encode.
        :return: DataFrame with encoded vectors in specified columns.
        """
        if not self.model:
            return df  # If no model is loaded, return the DataFrame as is

        def encode_text(words):
            """Encodes a list of words into Word2Vec vectors."""
            if isinstance(words, list):
                return np.array([self.model.wv[word] if word in self.model.wv else np.zeros(300) for word in words])
            return words  # Return unchanged if not a list (e.g., NaN or non-text values)

        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(encode_text)

        return df
