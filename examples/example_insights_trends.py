import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.trend_detector import TopicClusteringTrendDecorator

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Apply decorator
base_generator = BaseInsightGenerator()
trends_decorator = TopicClusteringTrendDecorator(base_generator)
insights = trends_decorator.extract_insights(df)

print(insights)
