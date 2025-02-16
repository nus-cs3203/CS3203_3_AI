from sentiment_analyser.classifier_strategy import SentimentClassifier


class SentimentAnalysisContext:
    """Context Class - Uses a selected strategy for sentiment analysis."""
    
    def __init__(self, strategy: SentimentClassifier):
        self.strategy = strategy  # Holds reference to a strategy

    def set_strategy(self, strategy: SentimentClassifier):
        """Dynamically switch sentiment analysis strategy."""
        self.strategy = strategy

    def analyze(self, text: str) -> dict:
        """Execute sentiment analysis using the current strategy."""
        return self.strategy.classify(text)
