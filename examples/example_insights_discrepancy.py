import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.post_analytics.sentiment_discrepancy_detector import SentimentDiscrepancyDecorator

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Apply detector
base_generator = BaseInsightGenerator()
discrepancy_detector = SentimentDiscrepancyDecorator(base_generator)

# Extract insights
insights = discrepancy_detector.extract_insights(df)
