import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.developer_analytics.cluster_maker import SentimentClusteringDecorator

df = pd.read_csv("files/all_complaints_2022_2025.csv")

base_generator = BaseInsightGenerator()
cluster_detector = SentimentClusteringDecorator(base_generator)

insights = cluster_detector.extract_insights(df)
