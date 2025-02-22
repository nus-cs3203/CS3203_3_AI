import re
from nltk.corpus import stopwords

from sentiment_analyser.sentiment_analysis_context import SentimentAnalysisContext

class SentimentPipelineBuilder:
    """Concrete Builder - Implements sentiment analysis pipeline with strategy support."""
    
    def __init__(self, strategy):
        self.context = SentimentAnalysisContext(strategy)  # Uses Context to handle strategies
        self.cleaned_text = None
        self.valid_text = None
        self.result = None

    def preprocess(self, text: str):
        """Text preprocessing: Lowercasing, removing special characters, and stopwords."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = text.split()
        words = [word for word in words if word not in stopwords.words('english')]
        self.cleaned_text = " ".join(words)
        return self
    
    def validate(self):
        """Text validation: Ensures the text is not empty after preprocessing."""
        if not self.cleaned_text.strip():
            self.valid_text = None
        else:
            self.valid_text = self.cleaned_text
        return self

    def analyze_sentiment(self):
        """Runs sentiment classification if validation passes."""
        if self.valid_text:
            self.result = self.context.analyze(self.valid_text)
        else:
            self.result = {"label": "neutral", "score": 0.0, "confidence": 0.0}  # Default neutral sentiment
        return self

    def get_result(self):
        """Returns the final sentiment result."""
        return self.result
