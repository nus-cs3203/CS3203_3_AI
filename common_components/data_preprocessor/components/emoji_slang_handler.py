import pandas as pd
import emoji
import re
import logging

class EmojiSlangHandler:
    """Handles slang conversion and emoji normalization in specific text columns of a DataFrame."""

    def __init__(self, text_columns, slang_dict=None):
        """
        :param text_columns: List of text columns to process.
        :param slang_dict: Optional custom dictionary for slang replacement.
        """
        self.text_columns = text_columns
        self.logger = logging.getLogger(__name__)

        # Allow a custom slang dictionary, or use the default one
        self.slang_dict = slang_dict or {
            "brb": "be right back",
            "lol": "laugh out loud",
            "omg": "oh my god",
            "idk": "i don't know",
            "btw": "by the way",
        }

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies slang replacement and emoji conversion to the specified text columns.

        :param df: Pandas DataFrame with text columns.
        :return: DataFrame with processed text columns.
        """
        if df.empty:
            self.logger.warning("DataFrame is empty. No slang or emoji processing needed.")
            return df

        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self._process_text)

        self.logger.info(f"Processed emoji and slang normalization for columns: {self.text_columns}")
        return df

    def _process_text(self, text):
        """Replaces slang and converts emojis for a single text entry."""
        if not isinstance(text, str):
            return text  # Return as-is if not a string

        # Tokenize while preserving punctuation (better than simple .split())
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)

        # Replace slang words efficiently
        tokens = [self.slang_dict.get(word.lower(), word) for word in tokens]

        # Convert emojis to text representation
        tokens = [emoji.demojize(word) for word in tokens]

        return " ".join(tokens)
