from sentiment_analyser.classifier_strategy import SentimentClassifier
from transformers import pipeline

import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')

class BERTSentimentClassifier(SentimentClassifier):
    """Concrete Strategy - Uses BERT for sentiment classification with confidence scoring."""
    
    def __init__(self):
        self.model = pipeline("sentiment-analysis")

    def classify(self, text: str) -> dict:
        result = self.model(text)[0]
        confidence = result["score"]  # Softmax probability as confidence
        return {"label": result["label"].lower(), "score": result["score"], "confidence": confidence}
