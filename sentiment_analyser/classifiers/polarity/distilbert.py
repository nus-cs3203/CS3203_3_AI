import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DistilBERTSentimentClassifier:
    """Uses DistilBERT for sentiment classification with -1 to 1 scaling on a DataFrame."""
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

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
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**tokens)

        logits = outputs.logits
        sentiment_score = torch.tanh(logits[0, 1] - logits[0, 0])  # Normalize to -1 to 1

        label = "positive" if sentiment_score >= 0 else "negative"

        return {
            "label": label,
            "score": sentiment_score.item()
        }
