import pandas as pd

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.prompt_decorator import PromptGeneratorDecorator
from insight_generator.sentiment_forecast_decorator import TopicSentimentForecastDecorator

# Load historical data
historical_data = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Load new posts
new_posts = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Custom column names
time_col = "created_utc"
sentiment_col = "sentiment_title_selftext_polarity"
category_col = "Domain Category"

# Step 1: Base Insight Generator
base_generator = BaseInsightGenerator()

# Step 2: Wrap with Sentiment Forecasting (Custom Columns)
forecast_decorator = TopicSentimentForecastDecorator(
    wrapped_insight_generator=base_generator,
    historical_data=historical_data,
    time_col=time_col,
    sentiment_col=sentiment_col,
    category_col=category_col,
    forecast_days=5  # Example: Forecast for next 5 days
)

# Step 3: Wrap with Poll Prompt Generator
prompt_generator = PromptGeneratorDecorator(forecast_decorator)

# Step 4: Process all posts
processed_posts = []
for _, post in new_posts.iterrows():
    insights = prompt_generator.extract_insights(post)
    processed_posts.append(insights)

# Convert to DataFrame
df_final = pd.DataFrame(processed_posts)

# Save output
df_final.to_csv("processed_reddit_insights.csv", index=False)

print("Processing completed. Insights saved to processed_reddit_insights.csv!")
