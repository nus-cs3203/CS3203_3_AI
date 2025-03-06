import pandas as pd

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator

# Sample historical sentiment data
historical_data = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Apply decorator
base_generator = BaseInsightGenerator()
forecast_decorator = TopicSentimentForecastDecorator(base_generator, historical_data)
insights = forecast_decorator.extract_insights(df)

print(insights)
