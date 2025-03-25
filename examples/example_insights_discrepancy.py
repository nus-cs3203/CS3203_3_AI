import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.developer_analytics.sentiment_discrepancy_detector import SentimentDiscrepancyDecorator

df = pd.read_csv("files/all_complaints_2022_2025.csv")

base_generator = BaseInsightGenerator()
discrepancy_detector = SentimentDiscrepancyDecorator(base_generator)

insights = discrepancy_detector.extract_insights(df)
