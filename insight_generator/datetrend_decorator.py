import pandas as pd
from insight_generator.base_decorator import InsightDecorator

class DateTrendTrackerDecorator(InsightDecorator):
    def __init__(self, wrapped_component, column_mappings=None, trend_window=7):
        """
        Tracks post trends over time by analyzing date-based activity.

        :param wrapped_component: The base insight generator to wrap.
        :param column_mappings: Dictionary for configurable column names.
            Example: { "date": "created_utc" }
        :param trend_window: Number of days to track trends (default: 7 days).
        """
        super().__init__(wrapped_component)
        self.column_mappings = column_mappings if column_mappings else {"date": "created_utc"}
        self.trend_window = trend_window  # Allows dynamic time window adjustments
        self.trend_data = []

    def track_trends(self, posts_df):
        """Track trends by counting posts per time range (e.g., daily, weekly)."""
        date_column = self.column_mappings["date"]

        if date_column not in posts_df:
            raise KeyError(f"Column '{date_column}' not found in DataFrame.")

        # Convert date column to datetime
        posts_df["date"] = pd.to_datetime(posts_df[date_column], unit='s', errors='coerce')

        # Filter out invalid dates
        posts_df = posts_df[posts_df["date"].notna()]

        # Convert to UTC (ensures consistency)
        posts_df["date"] = posts_df["date"].dt.tz_localize("UTC", ambiguous="NaT", errors="coerce")

        # Count posts per day
        daily_trends = posts_df.groupby(posts_df["date"].dt.date).size()
        return daily_trends

    def extract_insights(self, posts_df):
        """Override the base method to include date-related insights for batch processing."""
        insights_list = []
        
        # Track trends across the entire dataset
        trends = self.track_trends(posts_df)
        
        # Compute trends within the defined window
        trend_start_date = pd.Timestamp.utcnow().date() - pd.Timedelta(days=self.trend_window)
        filtered_trends = trends[trends.index > trend_start_date]
        trend_count = filtered_trends.sum()

        # Process each post
        for _, row in posts_df.iterrows():
            insights = super().extract_insights(row)  # Get existing insights from the wrapped component
            
            # Add trend insights
            insights[f"last_{self.trend_window}_days_post_count"] = trend_count
            self.trend_data.append(insights)
            insights_list.append(insights)
        
        return pd.DataFrame(insights_list)
    
    def generate_trend_report(self, report_path="trend_report.txt"):
        """Generate a separate report for post trends over time."""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Date-Based Trend Analysis ===\n\n")
            f.write(f"Tracking trends over the last {self.trend_window} days:\n")
            for entry in self.trend_data[:10]:  # Sample top 10 trend insights
                f.write(f"Post Count in Last {self.trend_window} Days: {entry[f'last_{self.trend_window}_days_post_count']}\n")
            f.write("\n" + "-"*50 + "\n\n")
        print(f"Date trend report saved to {report_path}")
