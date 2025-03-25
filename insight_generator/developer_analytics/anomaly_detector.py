from insight_generator.base_decorator import InsightDecorator
import pandas as pd
import numpy as np
import scipy.stats as stats

class SentimentAnomalyDetectionDecorator(InsightDecorator):
    def __init__(self, wrapped_insight_generator, historical_data, 
                 log_file="sentiment_anomalies.txt",
                 time_col="date", sentiment_col="sentiment_title_with_desc_score",
                 category_col="category", z_threshold=2.0):
        """
        Initializes the SentimentAnomalyDetectionDecorator.

        This decorator detects anomalies in sentiment trends by identifying sudden spikes or drops
        in sentiment scores for different categories.

        :param wrapped_insight_generator: The base insight generator to wrap.
        :param historical_data: A DataFrame containing historical sentiment data.
        :param log_file: File to log detected anomalies (default: "sentiment_anomalies.txt").
        :param time_col: Column name containing timestamps (default: "date").
        :param sentiment_col: Column name containing sentiment polarity scores 
                              (default: "sentiment_title_with_desc_score").
        :param category_col: Column name containing the topic/category (default: "category").
        :param z_threshold: Z-score threshold for flagging anomalies (default: 2.0).
        """
        super().__init__(wrapped_insight_generator)
        self.historical_data = historical_data
        self.log_file = log_file
        self.time_col = time_col
        self.sentiment_col = sentiment_col
        self.category_col = category_col
        self.z_threshold = z_threshold  # Threshold for detecting anomalies based on Z-scores.

    def extract_insights(self, df):
        """
        Extracts insights and detects anomalies in sentiment trends.

        :param df: DataFrame containing posts with sentiment data.
        :return: A dictionary of insights, including detected anomalies.
        """
        insights = self._wrapped.extract_insights(df)

        anomalies = {}

        # Group data by category and detect anomalies for each category.
        for category, group in df.groupby(self.category_col):
            anomaly_dates = self.detect_anomalies(category)
            if anomaly_dates:
                anomalies[category] = anomaly_dates

        insights["dates_with_shift"] = anomalies
        return insights

    def detect_anomalies(self, category):
        """
        Detects sentiment anomalies for a specific category.

        :param category: The category/topic to analyze.
        :return: A list of dates where anomalies were detected.
        """
        # Filter historical data for the given category.
        topic_data = self.historical_data[self.historical_data[self.category_col] == category].copy()

        if topic_data.empty:
            return []

        # Convert timestamps to datetime and ensure timezone-naive format.
        topic_data["ds"] = pd.to_datetime(topic_data[self.time_col], errors="coerce", utc=True).dt.tz_localize(None)

        # Group data by day and compute the mean sentiment score for each day.
        daily_sentiment = topic_data.groupby(topic_data["ds"].dt.date)[self.sentiment_col].mean()

        if len(daily_sentiment) < 5:
            return []  # Not enough data points for reliable anomaly detection.

        # Compute Z-scores for daily sentiment values.
        if daily_sentiment.std() == 0:
            return []  # Avoid division by zero if standard deviation is zero.

        z_scores = stats.zscore(daily_sentiment)

        # Identify dates where the absolute Z-score exceeds the threshold.
        anomaly_dates = daily_sentiment.index[np.abs(z_scores) > self.z_threshold]
        anomaly_dates = [str(date) for date in anomaly_dates]

        return anomaly_dates