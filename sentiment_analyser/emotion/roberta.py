import pandas as pd
from transformers import pipeline
from sentiment_analyser.strategy import SentimentClassifier  # Ensuring it follows Strategy Pattern

class RobertaClassifier(SentimentClassifier):
    """Concrete Strategy - Uses DistilBERT for fine-grained emotion classification on DataFrame columns."""
    
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions"):
        self.model = pipeline("text-classification", model=model_name, return_all_scores=True)

    def classify(self, df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
        df = df.copy()  # Avoid modifying the original DataFrame

        for col in text_cols:
            results = df[col].astype(str).apply(lambda text: self._predict_emotion(text))
            
            df[col + "_emotion"] = results.apply(lambda r: r["emotion"])

        return df
    
    def _predict_emotion(self, text: str) -> dict:
        scores = self.model(text)[0]
        top_emotion = max(scores, key=lambda x: x["score"])["label"]  # Get highest scoring emotion

        return {
            "emotion": top_emotion
        }
