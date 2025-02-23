import pandas as pd
import emoji

class EmojiSlangHandler:
    """Handles slang conversion and emoji normalization in specific text columns of a DataFrame."""

    def __init__(self, text_columns):
        """
        :param text_columns: List of text columns to process.
        """
        self.text_columns = text_columns
        self.slang_dict = {
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
        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._process_text)
        return df

    def _process_text(self, text):
        """Replaces slang and converts emojis for a single text entry."""
        if isinstance(text, str):
            words = text.split()  # Tokenize based on spaces
            words = [self.slang_dict.get(word.lower(), word) for word in words]  # Replace slang
            words = [emoji.demojize(word) for word in words]  # Convert emojis to text
            return " ".join(words)
        return text  # Return as-is if not a string
