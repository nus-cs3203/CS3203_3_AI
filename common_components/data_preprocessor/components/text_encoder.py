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
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        if model_path:
            try:
                self.model = Word2Vec.load(model_path)
                self.vector_size = self.model.vector_size
                self.logger.info(f"Loaded Word2Vec model from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load Word2Vec model from {model_path}: {e}")
                self.model = None
                self.vector_size = 300  # Default size if no model is loaded
        else:
            self.model = None
            self.vector_size = 300  # Default size if no model is loaded
            self.logger.info("No Word2Vec model path provided. Using default vector size.")

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
                try:
                    df[col] = df[col].apply(self._encode_text)
                    self.logger.info(f"Encoded column '{col}' successfully.")
                except Exception as e:
                    self.logger.error(f"Error encoding column '{col}': {e}")
            else:
                self.logger.warning(f"Column '{col}' not found in DataFrame.")

        return df

    def _encode_text(self, text):
        """Encodes a text entry into a fixed-size vector using Word2Vec."""
        if isinstance(text, str):
            words = word_tokenize(text.lower())  # Tokenize and lowercase
            word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]

            if word_vectors:
                return np.mean(word_vectors, axis=0)  # Return mean word vector
            else:
                self.logger.warning("No valid words found in text. Returning zero vector.")
                return np.zeros(self.vector_size)  # Return zero vector if no valid words

        self.logger.warning("Input is not a string. Returning zero vector.")
        return np.zeros(self.vector_size)  # Return zero vector if input is not a string
