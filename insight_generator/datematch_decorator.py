import pandas as pd
from insight_generator.base_decorator import InsightDecorator

class DateMatcherDecorator(InsightDecorator):
    def __init__(self, wrapped_component):
        super().__init__(wrapped_component)

    def track_trends(self, posts_df):
        """Track trends by counting posts per time range (e.g., daily, weekly)"""
        # Convert 'created_utc' to datetime
        posts_df['date'] = pd.to_datetime(posts_df['created_utc'], unit='s')
        
        # Count posts per day
        daily_trends = posts_df.groupby(posts_df['date'].dt.date).size()
        return daily_trends

    def extract_insights(self, post):
        """Override the base method to include date-related insights"""
        insights = super().extract_insights(post)  # Get existing insights from the wrapped component
        
        # You can optionally pass a DataFrame of posts and get trends, for simplicity:
        trends = self.track_trends(post)
        
        # Here we will just include the count of posts in the last week (or similar)
        last_week_trends = trends[trends.index > (pd.Timestamp.today() - pd.Timedelta(7, 'D'))]
        insights["last_week_post_count"] = len(last_week_trends)
        
        return insights
