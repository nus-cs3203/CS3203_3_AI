from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
import os
import re
from sentiment_analyser.strategy import SentimentClassifier
from transformers import pipeline

class CustomSentimentClassifier(SentimentClassifier):

    def __init__(self):
        self.model = SentimentIntensityAnalyzer()
        self.classifier = pipeline("sentiment-analysis", model="common_components/singlish_classifier_2", truncation=True, max_length=512)
        
    def classify(self, df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
        """Applies sentiment analysis to specified columns in a Pandas DataFrame."""
        df = df.copy()
        
        for col in text_cols:
            df[col + "_classifier_results"] = df[col].astype(str).apply(self._analyze_text_classifier)
            df[col + "_score"] = df.apply(lambda row: row[col + "_classifier_results"]["score"], axis=1)
            df[col + "_label"] = df[col + "_score"].apply(self._determine_label)
            df.drop(columns=[col + "_classifier_results"], inplace=True)
        
        return df

    def _determine_label(self, score):
        if score >= 0.1:
            return "positive"
        elif score <= -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_text_classifier(self, text: str) -> dict:
        """Helper function to classify sentiment of a single text input using a custom classifier."""
        result = self.classifier(text)[0]
        sentiment_score = result["score"] if result["label"] == "positive" else -result["score"]
        
        return {"label": result["label"], "score": sentiment_score}
