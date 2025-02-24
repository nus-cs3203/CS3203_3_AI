from insight_generator.base_decorator import InsightDecorator

class ImportanceScoringDecorator(InsightDecorator):
    def extract_insights(self, post):
        insights = super().extract_insights(post)
        insights["importance"] = self.calculate_importance(insights)
        return insights

    def calculate_importance(self, insights):
        score = insights["engagement"]["score"]
        sentiment = insights["sentiment"]
        sentiment_weight = 10 if sentiment == "negative" else 5 if sentiment == "neutral" else 1
        return score * sentiment_weight
    
    def process_batch(self, posts_df):
        """Handle a batch of posts in DataFrame"""
        return posts_df.apply(self.extract_insights, axis=1)
