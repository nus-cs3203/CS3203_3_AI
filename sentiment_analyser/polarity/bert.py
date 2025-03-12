import pandas as pd
from transformers import pipeline
from sentiment_analyser.strategy import SentimentClassifier  # Follows Strategy Pattern

class BERTClassifier(SentimentClassifier):
    """Concrete Strategy - Uses DistilBERT for sentiment classification on DataFrame columns."""
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.model = pipeline("text-classification", model=model_name, truncation=True)

    def classify(self, df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
        df = df.copy()  # Avoid modifying the original DataFrame

        for col in text_cols:
            results = df[col].astype(str).apply(lambda text: self._analyze_text(text))
            
            df[col + "_label"] = results.apply(lambda r: r["label"])
            df[col + "_score"] = results.apply(lambda r: r["score"])  # Now scaled from -1 to 1

        return df
    
    def _analyze_text(self, text: str) -> dict:
        scores = self.model(text)[0]  # Get sentiment result
        sentiment_label = scores["label"].lower()  # Convert label to lowercase
        confidence = scores["score"]  # Model's probability score (0-1)

        # Convert confidence score to polarity (-1 to 1)
        polarity_score = confidence if sentiment_label == "positive" else -confidence

        return {
            "label": sentiment_label,
            "score": polarity_score  # Now correctly scaled from -1 to 1
        }
