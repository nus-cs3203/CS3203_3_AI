import pandas as pd
import matplotlib.pyplot as plt

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator

# Load and preprocess historical data
historical_data = pd.read_csv("files/all_complaints_2022_2025.csv").head(1000)
historical_data["date"] = pd.to_datetime(historical_data["date"], errors="coerce")
aggregated_data = historical_data.groupby([pd.Grouper(key="date", freq="m"), "category"])["sentiment"].mean().reset_index()


# Load Reddit posts and check category counts
df = pd.read_csv("files/all_complaints_2022_2025.csv").tail(1000)
print(df['category'].value_counts())

# Generate forecasted insights
base_generator = BaseInsightGenerator()
forecast_decorator = TopicSentimentForecastDecorator(base_generator)
forecast_insights = forecast_decorator.extract_insights(df)

print(forecast_insights)
