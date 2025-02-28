import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import logging

class TextEncoder:
    """Encodes text columns using a Word2Vec model."""

    def __init__(self, text_columns, model_path=None):
        """
        Initializes the TextEncoder with an optional pre-trained Word2Vec model.

        :param text_columns: List of text columns to process.
        :param model_path: Path to a pre-trained Word2Vec model (optional).
        """
        self.text_columns = text_columns
        self.logger = logging.getLogger(__name__)
        
        if model_path:
            self.model = Word2Vec.load(model_path)
            self.vector_size = self.model.vector_size
        else:
            self.model = None
            self.vector_size = 300  # Default size if no model is loaded

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes text in the specified columns using Word2Vec embeddings.

        :param df: Pandas DataFrame containing text data.
        :return: DataFrame with encoded text as fixed-size vectors.
        """
        if not self.model:
            self.logger.warning("No Word2Vec model loaded. Returning DataFrame as is.")
            return df

        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._encode_text)

        return df

    def _encode_text(self, text):
        """Encodes a text entry into a fixed-size vector using Word2Vec."""
        if isinstance(text, str):
            words = word_tokenize(text.lower())  # Tokenize and lowercase
            word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]

            if word_vectors:
                return np.mean(word_vectors, axis=0)  # Return mean word vector
            else:
                return np.zeros(self.vector_size)  # Return zero vector if no valid words

        return np.zeros(self.vector_size)  # Return zero vector if input is not a string
