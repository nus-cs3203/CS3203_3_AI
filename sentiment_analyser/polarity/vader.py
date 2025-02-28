from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
from sentiment_analyser.strategy import SentimentClassifier

nltk.download("vader_lexicon")

class VaderSentimentClassifier(SentimentClassifier):
    """Concrete Strategy - Uses VADER for sentiment classification on DataFrame columns."""
    
    def __init__(self):
        self.model = SentimentIntensityAnalyzer()

    def classify(self, df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
        """
        Applies sentiment analysis to specified columns in a Pandas DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_cols (list): List of column names to perform sentiment analysis on.

        Returns:
            pd.DataFrame: DataFrame with additional sentiment-related columns.
        """
        df = df.copy()  # Avoid modifying the original DataFrame

        for col in text_cols:
            results = df[col].astype(str).apply(self._analyze_text)
            df[col + "_score"] = results.apply(lambda r: r["score"])
            df[col + "_label"] = results.apply(lambda r: r["label"])

        return df

    def _analyze_text(self, text: str) -> dict:
        """Helper function to classify sentiment of a single text input."""
        scores = self.model.polarity_scores(text)
        sentiment_score = scores["compound"]  # VADER's compound score (-1 to 1)

        # Define sentiment labels
        if sentiment_score >= 0.05:
            label = "positive"
        elif sentiment_score <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": sentiment_score
        }
