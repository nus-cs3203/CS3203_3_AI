from insight_generator.base_decorator import InsightDecorator
import pandas as pd
import numpy as np
import scipy.stats as stats

class SentimentAnomalyDetectionDecorator(InsightDecorator):
    def __init__(self, wrapped_insight_generator, historical_data, 
                 log_file="sentiment_anomalies.txt",
                 time_col="date", sentiment_col="sentiment_title_with_description_score",
                 category_col="category", z_threshold=2.0):
        """
        Detects anomalies in sentiment trends by identifying sudden spikes or drops.
        :param wrapped_insight_generator: Base insight generator to wrap
        :param historical_data: A DataFrame containing historical sentiment data
        :param log_file: File to log detected anomalies
        :param time_col: Column containing timestamps
        :param sentiment_col: Column containing sentiment polarity scores
        :param category_col: Column containing the topic/category
        :param z_threshold: Z-score threshold for flagging anomalies
        """
        super().__init__(wrapped_insight_generator)
        self.historical_data = historical_data
        self.log_file = log_file
        self.time_col = time_col
        self.sentiment_col = sentiment_col
        self.category_col = category_col
        self.z_threshold = z_threshold  # Defines how extreme a change must be to be flagged

    def extract_insights(self, df):
        """
        Extracts insights including anomaly detection for sentiment trends.
        :param df: DataFrame containing posts
        :return: Insights with anomaly detection results
        """
        insights = self._wrapped.extract_insights(df)

        anomalies = {}

        for category, group in df.groupby(self.category_col):
            anomaly_dates = self.detect_anomalies(category)
            if anomaly_dates:
                anomalies[category] = anomaly_dates

        insights["dates_with_shift"] = anomalies
        return insights

    def detect_anomalies(self, category):
        """
        Detects sentiment shifts for a given category.
        :param category: The category/topic to analyze
        :return: List of dates where anomalies were detected
        """
        # Filter historical data for this category
        topic_data = self.historical_data[self.historical_data[self.category_col] == category].copy()

        if topic_data.empty:
            return []

        # Convert timestamps, ensuring timezone-naive format
        topic_data["ds"] = pd.to_datetime(topic_data[self.time_col], errors="coerce", utc=True).dt.tz_localize(None)

        # Group by day and compute mean sentiment
        daily_sentiment = topic_data.groupby(topic_data["ds"].dt.date)[self.sentiment_col].mean()

        if len(daily_sentiment) < 5:
            return []  # Not enough data for anomaly detection

        # Compute Z-scores with a safety check
        if daily_sentiment.std() == 0:
            return []  # Avoid division by zero errors

        z_scores = stats.zscore(daily_sentiment)

        # Flag anomalies where |Z-score| > threshold
        anomaly_dates = daily_sentiment.index[np.abs(z_scores) > self.z_threshold]
        anomaly_dates = [str(date) for date in anomaly_dates]

        return anomaly_dates