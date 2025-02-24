import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DistilBERTSentimentClassifier:
    """Uses DistilBERT for sentiment classification with -1 to 1 scaling."""
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def classify(self, text: str) -> dict:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**tokens)
        
        logits = outputs.logits
        sentiment_score = torch.tanh(logits[0, 1] - logits[0, 0])  # Normalize to -1 to 1

        label = "positive" if sentiment_score >= 0 else "negative"

        return {
            "label": label,
            "score": sentiment_score.item()
        }
