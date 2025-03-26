from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
import os
import re
from sentiment_analyser.strategy import SentimentClassifier

class VaderSentimentClassifier(SentimentClassifier):
    """Uses VADER (or a custom lexicon) for sentiment classification on DataFrame columns."""

    def __init__(self, lexicon_file="files/singlish_lexicon_cleaned.csv"):
        self.lexicon_file = lexicon_file
        self.model = SentimentIntensityAnalyzer()
        
        if lexicon_file and os.path.exists(lexicon_file):
            lexicon_dict = self._load_custom_lexicon(lexicon_file)
            self.model.lexicon.update(lexicon_dict)

    def _load_custom_lexicon(self, lexicon_file):
        """Loads a custom lexicon."""
        lexicon_df = pd.read_csv(lexicon_file)
        required_columns = ["singlish", "sentiment_score"]
        if not all(col in lexicon_df.columns for col in required_columns):
            raise ValueError(f"Custom lexicon must have columns: {required_columns}")
        
        return {row["singlish"]: row["sentiment_score"] for _, row in lexicon_df.iterrows()}

    def classify(self, df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
        """Applies sentiment analysis to specified DataFrame columns."""
        df = df.copy()
        
        for col in text_cols:
            results = df[col].astype(str).apply(self._analyze_text)
            df[col + "_score"] = results.apply(lambda r: r["score"])
            df[col + "_label"] = results.apply(lambda r: r["label"])
        
        return df
    
    def _analyze_text(self, text: str) -> dict:
        """Performs sentiment analysis."""
        text = self._adjust_negations(text)
        scores = self.model.polarity_scores(text)
        sentiment_score = scores["compound"]
        
        if sentiment_score >= 0.1:
            label = "positive"
        elif sentiment_score <= -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return {"label": label, "score": sentiment_score}
    
    def _adjust_negations(self, text: str) -> str:
        """Handles negations."""
        negations = ["not", "isn't", "wasn't", "can't", "couldn't"]
        for neg in negations:
            text = re.sub(rf"\b{neg} ([a-zA-Z]+)\b", r"less \1", text, flags=re.IGNORECASE)
        return text
