import pandas as pd
import matplotlib.pyplot as plt

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator

# Load historical data
historical_data = pd.read_csv("files/2022_2025_merged.csv")

# Convert to datetime
historical_data["date"] = pd.to_datetime(historical_data["date"], errors="coerce")

# Group by week and category, then aggregate sentiment scores
aggregated_data = historical_data.groupby([pd.Grouper(key="date", freq="m"), "category"])["sentiment"].mean().reset_index()

# Plot sentiment trends
plt.figure(figsize=(12, 6))
categories = aggregated_data["category"].unique()
for category in categories:
    category_data = aggregated_data[aggregated_data["category"] == category]
    
    if len(category_data) < 10:
        continue  # Skip categories with too few data points

    plt.plot(category_data["date"], category_data["sentiment"], label=category)

plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.title("Sentiment Trends by Category")
plt.legend()
plt.show()

# Sample Reddit posts
df = pd.read_csv("files/all_complaints_2022_2025.csv")
# Check if there is data for each category
category_counts = df['category'].value_counts()
print(category_counts)

# Apply decorators
base_generator = BaseInsightGenerator()

# Forecasted insights
forecast_decorator = TopicSentimentForecastDecorator(base_generator)  # Pass instance, not class
forecast_insights = forecast_decorator.extract_insights(df)

print(forecast_insights)
