import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.developer_analytics.cluster_maker import SentimentClusteringDecorator

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Apply detector
base_generator = BaseInsightGenerator()
cluster_detector = SentimentClusteringDecorator(base_generator)

# Extract insights
insights = cluster_detector.extract_insights(df)
