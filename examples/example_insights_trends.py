import pandas as pd
from insight_generator.base_insight_developer import BaseInsightDeveloperGenerator
from insight_generator.developer_analytics.trend_detector import KeywordsTrendDecorator

df = pd.read_csv("files/all_complaints_2022_2025.csv").head(1000)  # Load sample data

base_generator = BaseInsightDeveloperGenerator()
trends_decorator = KeywordsTrendDecorator(base_generator)
insights = trends_decorator.extract_insights(df)

print(insights)
