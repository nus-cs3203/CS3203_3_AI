from nltk.sentiment import SentimentIntensityAnalyzer
from sentiment_analyser.classifier_strategy import SentimentClassifier
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')


class VaderSentimentClassifier(SentimentClassifier):
    """Concrete Strategy - Uses VADER for sentiment classification with confidence scoring."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def classify(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)
        sentiment = "neutral"
        if scores["compound"] >= 0.05:
            sentiment = "positive"
        elif scores["compound"] <= -0.05:
            sentiment = "negative"
        
        confidence = abs(scores["compound"])  # Higher magnitude = stronger sentiment
        return {"label": sentiment, "score": scores["compound"], "confidence": confidence}
