import pandas as pd
from insight_generator.base_decorator import InsightDecorator

class EngagementDecorator(InsightDecorator):
    def __init__(self, wrapped_component, column_mappings=None):
        """
        Adds engagement insights by calculating scores based on votes, comments, and sentiment.

        :param wrapped_component: The base insight generator to wrap.
        :param column_mappings: Dictionary for configurable column names.
        """
        super().__init__(wrapped_component)
        self.column_mappings = column_mappings or {
            "upvotes": "ups",
            "downvotes": "downs",
            "score": "score",
            "upvote_ratio": "upvote_ratio",
            "num_comments": "num_comments",
            "sentiment_title": "sentiment_title_selftext_polarity",
            "sentiment_comments": "sentiment_comments_polarity",
        }

    def extract_insights(self, post):
        """Extracts engagement-based insights for a single post."""
        insights = super().extract_insights(post)  # Get base insights
        
        upvotes = post.get(self.column_mappings["upvotes"], 0)
        downvotes = post.get(self.column_mappings["downvotes"], 0)
        num_comments = post.get(self.column_mappings["num_comments"], 0)

        # Compute engagement score
        engagement_score = (upvotes * 0.7 + num_comments * 0.3) - downvotes

        # Sentiment impact
        sentiment_title = post.get(self.column_mappings["sentiment_title"], 0)
        sentiment_comments = post.get(self.column_mappings["sentiment_comments"], 0)
        valid_sentiments = [s for s in [sentiment_title, sentiment_comments] if s is not None]
        avg_sentiment = sum(valid_sentiments) / len(valid_sentiments) if valid_sentiments else 0

        sentiment_influence = "Neutral"
        if avg_sentiment > 0:
            sentiment_influence = "Positive"
            engagement_score *= 1.2  # Boost for positive sentiment
        elif avg_sentiment < 0:
            sentiment_influence = "Negative"
            engagement_score *= 0.8  # Reduce for negative sentiment

        insights["engagement_score"] = engagement_score
        insights["sentiment_influence"] = sentiment_influence

        return insights

    def process_batch(self, posts_df):
        """Processes a batch of posts and adds engagement insights as new columns."""
        insights_df = posts_df.apply(self.extract_insights, axis=1, result_type="expand")
        return posts_df.assign(**insights_df)  # Directly add new columns
