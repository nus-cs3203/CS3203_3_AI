import pandas as pd
from insight_generator.base_insight_developer import BaseInsightDeveloperGenerator
from insight_generator.developer_analytics.sentiment_discrepancy_detector import SentimentDiscrepancyDecorator

df = pd.read_csv("files/sentiment_scored_2023_data.csv")

if "domain_category" in df.columns:
    df.rename(columns={"domain_category": "category"}, inplace=True)

if "Domain Category" in df.columns:
    df.rename(columns={"Domain Category": "category"}, inplace=True)

base_generator = BaseInsightDeveloperGenerator()
discrepancy_detector = SentimentDiscrepancyDecorator(base_generator, score_col_1="sentiment_title_selftext_polarity", score_col_2="sentiment_comments_polarity")

insights = discrepancy_detector.extract_insights(df)
print(insights)
