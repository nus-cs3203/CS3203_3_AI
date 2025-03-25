import pandas as pd
import matplotlib.pyplot as plt

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator

# Load and preprocess historical data
historical_data = pd.read_csv("files/2022_2025_merged.csv")
historical_data["date"] = pd.to_datetime(historical_data["date"], errors="coerce")
aggregated_data = historical_data.groupby([pd.Grouper(key="date", freq="m"), "category"])["sentiment"].mean().reset_index()

# Plot sentiment trends
plt.figure(figsize=(12, 6))
categories = aggregated_data["category"].unique()
for category in categories:
    category_data = aggregated_data[aggregated_data["category"] == category]
    if len(category_data) < 10:
        continue
    plt.plot(category_data["date"], category_data["sentiment"], label=category)

plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.title("Sentiment Trends by Category")
plt.legend()
plt.show()

# Load Reddit posts and check category counts
df = pd.read_csv("files/all_complaints_2022_2025.csv")
print(df['category'].value_counts())

# Generate forecasted insights
base_generator = BaseInsightGenerator()
forecast_decorator = TopicSentimentForecastDecorator(base_generator)
forecast_insights = forecast_decorator.extract_insights(df)

print(forecast_insights)
