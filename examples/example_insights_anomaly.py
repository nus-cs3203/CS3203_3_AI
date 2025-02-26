import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.anomaly_detector import SentimentAnomalyDetectionDecorator

# Sample historical sentiment data
historical_data = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Apply anomaly detector
base_generator = BaseInsightGenerator()
anomaly_detector = SentimentAnomalyDetectionDecorator(base_generator, historical_data)

# Extract insights
insights = anomaly_detector.extract_insights(df)

print(insights["sentiment_anomalies"])
