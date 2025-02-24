from insight_generator.base_decorator import InsightDecorator

class EngagementDecorator(InsightDecorator):
    def extract_insights(self, post):
        # Extract base insights from the wrapped generator
        insights = super().extract_insights(post)
        
        # Engagement metrics
        upvotes = post["ups"]
        downvotes = post["downs"]
        score = post["score"]
        upvote_ratio = post["upvote_ratio"]
        num_comments = post["num_comments"]
        
        # Calculating engagement score (simple example: positive correlation with upvotes and comments)
        engagement_score = (upvotes * 0.7 + num_comments * 0.3) - downvotes
        
        # Adding engagement insights
        insights["engagement"] = {
            "upvotes": upvotes,
            "downvotes": downvotes,
            "score": score,
            "upvote_ratio": upvote_ratio,
            "num_comments": num_comments,
            "engagement_score": engagement_score  # Derived score
        }

        # Sentiment-based analysis for engagement
        sentiment_title = post["sentiment_title_selftext_polarity"]
        sentiment_comments = post["sentiment_comments_polarity"]
        avg_sentiment = (sentiment_title + sentiment_comments) / 2
        
        # Sentiment-driven analysis: if positive sentiment, engagement score might be higher
        if avg_sentiment > 0:
            insights["engagement"]["sentiment_influence"] = "Positive"
            insights["engagement"]["adjusted_engagement_score"] = engagement_score * 1.2  # Boost for positive sentiment
        else:
            insights["engagement"]["sentiment_influence"] = "Negative"
            insights["engagement"]["adjusted_engagement_score"] = engagement_score * 0.8  # Reduce for negative sentiment

        return insights
    
    def process_batch(self, posts_df):
        """Handle a batch of posts in DataFrame"""
        return posts_df.apply(self.extract_insights, axis=1)
