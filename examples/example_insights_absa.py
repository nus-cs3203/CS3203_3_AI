import pandas as pd

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.category_absa_finder import CategoryABSAInsightDecorator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv").head(100)
df.dropna(subset=["title"], inplace=True)

# Apply decorator
base_generator = BaseInsightGenerator()
absa_decorator = CategoryABSAInsightDecorator(base_generator)
insights = absa_decorator.extract_insights(df)

print(insights)
