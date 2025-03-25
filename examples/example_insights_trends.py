import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.developer_analytics.trend_detector import KeywordsTrendDecorator

# Sample Reddit posts
df = pd.read_csv("files/all_complaints_2022_2025.csv").head(1000)

# Apply decorator
base_generator = BaseInsightGenerator()
trends_decorator = KeywordsTrendDecorator(base_generator)
insights = trends_decorator.extract_insights(df)

print(insights)
