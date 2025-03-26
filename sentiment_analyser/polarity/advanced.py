from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
import os
import re
from sentiment_analyser.strategy import SentimentClassifier
from transformers import pipeline

class AdvancedSentimentClassifier(SentimentClassifier):

    def __init__(self, lexicon_file="files/singlish_lexicon_cleaned.csv"):
        self.lexicon_file = lexicon_file
        self.model = SentimentIntensityAnalyzer()
        self.classifier = pipeline("sentiment-analysis", model="common_components/singlish_classifier_2", truncation=True, max_length=512)
        
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
        """Applies sentiment analysis to specified columns in a DataFrame."""
        df = df.copy()
        
        for col in text_cols:
            df[col + "_vader_results"] = df[col].astype(str).apply(self._analyze_text_vader)
            df[col + "_classifier_results"] = df[col].astype(str).apply(self._analyze_text_classifier)
            df[col + "_score"] = df.apply(lambda row: (row[col + "_vader_results"]["score"] + row[col + "_classifier_results"]["score"]) / 2, axis=1)
            df[col + "_label"] = df[col + "_score"].apply(self._determine_label)
            df.drop(columns=[col + "_vader_results", col + "_classifier_results"], inplace=True)
        
        return df

    def _determine_label(self, score):
        if score >= 0.1:
            return "positive"
        elif score <= -0.1:
            return "negative"
        else:
            return "neutral"

    def _analyze_text_vader(self, text: str) -> dict:
        """Classifies sentiment using VADER."""
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
    
    def _analyze_text_classifier(self, text: str) -> dict:
        """Classifies sentiment using a custom classifier."""
        result = self.classifier(text)[0]
        sentiment_score = result["score"] if result["label"] == "positive" else -result["score"]
        
        return {"label": result["label"], "score": sentiment_score}

    def _adjust_negations(self, text: str) -> str:
        """Handles negations."""
        negations = ["not", "isn't", "wasn't", "can't", "couldn't"]
        for neg in negations:
            text = re.sub(rf"\b{neg} ([a-zA-Z]+)\b", r"less \1", text, flags=re.IGNORECASE)
        return text
