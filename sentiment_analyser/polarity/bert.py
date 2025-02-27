from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

class BERTClassifier:
    """Concrete Strategy - Uses BERT logits for sentiment scoring between -1 to 1."""
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16)

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
