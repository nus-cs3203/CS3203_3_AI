from abc import ABC, abstractmethod

class SentimentClassifier(ABC):
    """Strategy Interface - Defines a method for sentiment classification with confidence scoring."""
    
    @abstractmethod
    def classify(self, text: str) -> dict:
        """Takes text input and returns sentiment label, score, and confidence level."""
        pass
