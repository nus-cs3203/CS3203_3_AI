import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class XLMRSentimentClassifier:
    """Uses XLM-RoBERTa for sentiment analysis with -1 to 1 scores."""
    
    def __init__(self, model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def classify(self, df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
        """Classifies sentiment of text columns in a DataFrame."""
        df = df.copy()  # Avoid modifying original DataFrame

        for col in text_cols:
            results = df[col].astype(str).apply(self.classify_text)
            df[col + "_score"] = results.apply(lambda r: r["score"])
            df[col + "_label"] = results.apply(lambda r: r["label"])

        return df
