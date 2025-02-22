from insight_generator.base_decorator import InsightDecorator

class AggregatorDecorator(InsightDecorator):
    def __init__(self, wrapped_insight_generator, sentiment_threshold=0.5):
        """
        Aggregates title and comment-level sentiment and highlights significant differences.
        :param wrapped_insight_generator: The base insight generator to wrap
        :param sentiment_threshold: The threshold for sharp sentiment differences
        """
        super().__init__(wrapped_insight_generator)
        self.sentiment_threshold = sentiment_threshold

    def extract_insights(self, post):
        insights = super().extract_insights(post)
        
        # Extract title sentiment and comment sentiment
        title_sentiment_score = post.get("sentiment_title_selftext_polarity", 0)
        comment_sentiment_scores = post.get("sentiment_comments_polarity", [])
        
        # If there are comments, calculate their average sentiment score
        if comment_sentiment_scores:
            comment_sentiment_score = sum(comment_sentiment_scores) / len(comment_sentiment_scores)
        else:
            comment_sentiment_score = 0
        
        # Combine both title and comment sentiment scores
        insights["combined_sentiment"] = {
            "title_sentiment": title_sentiment_score,
            "comment_sentiment": comment_sentiment_score,
            "sentiment_diff": abs(title_sentiment_score - comment_sentiment_score)
        }
        
        # Highlight areas where the difference is sharp
        if insights["combined_sentiment"]["sentiment_diff"] > self.sentiment_threshold:
            insights["sharp_diff_alert"] = True
        else:
            insights["sharp_diff_alert"] = False
        
        return insights
