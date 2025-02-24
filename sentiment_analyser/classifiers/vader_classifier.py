from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
from sentiment_analyser.strategy import SentimentClassifier

nltk.download('vader_lexicon')

class VaderSentimentClassifier(SentimentClassifier):
    """Concrete Strategy - Uses VADER for sentiment classification on DataFrame columns."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def classify(self, df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
        df = df.copy()  # Avoid modifying the original DataFrame

        for col in text_cols:
            results = df[col].astype(str).apply(lambda text: self.analyzer.polarity_scores(text))
            df[col + "_score"] = results.apply(lambda r: r["compound"])
            df[col + "_label"] = results.apply(lambda r: "positive" if r["compound"] >= 0.05 else "negative" if r["compound"] <= -0.05 else "neutral")

        return df
