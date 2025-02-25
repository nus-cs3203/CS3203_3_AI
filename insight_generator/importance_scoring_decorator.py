import pandas as pd
from insight_generator.base_decorator import InsightDecorator

class ImportanceScoringDecorator(InsightDecorator):
    def __init__(self, wrapped_component, sentiment_weights=None):
        """
        Enhances insights by assigning an importance score based on engagement and sentiment.

        :param wrapped_component: The base insight generator to wrap.
        :param sentiment_weights: Dictionary defining weights for different sentiment types.
            Example: { "negative": 10, "neutral": 5, "positive": 1 }
        """
        super().__init__(wrapped_component)
        self.sentiment_weights = sentiment_weights or { "negative": 10, "neutral": 5, "positive": 1 }

    def extract_insights(self, post):
        """Extracts importance scoring insights for a single post."""
        insights = super().extract_insights(post)

        # Compute importance score
        importance_score = self.calculate_importance(insights)
        insights["importance_score"] = importance_score  # Store importance score

        return insights

    def calculate_importance(self, insights):
        """Calculates importance score based on engagement and sentiment."""
        # Retrieve engagement score, default to 0 if missing
        engagement_score = insights.get("engagement_score", 0)

        # Retrieve sentiment influence and determine weight
        sentiment_influence = insights.get("sentiment_influence", "neutral").lower()
        sentiment_weight = self.sentiment_weights.get(sentiment_influence, 5)  # Default weight = 5

        return engagement_score * sentiment_weight

    def process_batch(self, posts_df):
        """Processes a batch of posts and adds the importance score as a new column."""
        importance_scores = posts_df.apply(lambda row: self.extract_insights(row)["importance_score"], axis=1)
        return posts_df.assign(importance_score=importance_scores)
