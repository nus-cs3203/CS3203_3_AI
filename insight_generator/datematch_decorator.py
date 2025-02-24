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

    def extract_insights(self, posts_df):
        """Override the base method to include date-related insights for batch processing"""
        insights_list = []
        
        # Track trends across the entire DataFrame
        trends = self.track_trends(posts_df)
        
        # Process each post
        for _, row in posts_df.iterrows():
            insights = super().extract_insights(row)  # Get existing insights from the wrapped component
            
            # Filter trends for the last week
            last_week_trends = trends[trends.index > (pd.Timestamp.today() - pd.Timedelta(7, 'D'))]
            insights["last_week_post_count"] = len(last_week_trends)
            
            insights_list.append(insights)
        
        return pd.DataFrame(insights_list)
