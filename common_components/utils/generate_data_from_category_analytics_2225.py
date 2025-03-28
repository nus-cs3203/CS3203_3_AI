import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator
from insight_generator.category_analytics.llm_category_absa import CategoryABSAWithLLMInsightDecorator
from insight_generator.category_analytics.llm_category_summarizer import CategorySummarizerDecorator

# Sample Reddit posts

historical_data = pd.read_csv("files/2022_2025_merged.csv")
historical_data['date'] = pd.to_datetime(historical_data['date'])
historical_data.dropna(subset=["title"], inplace=True)
# Filter historical data for 2022-2024
historical_data = historical_data[(historical_data['date'].dt.year >= 2022) & (historical_data['date'].dt.year <= 2024)]

# Update df to have only 2025 data
df = pd.read_csv("files/2022_2025_merged.csv")
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year == 2025]
df.dropna(subset=["title"], inplace=True)


# Apply decorators
base_generator = BaseInsightGenerator()

# Forecasted insights
forecast_decorator = TopicSentimentForecastDecorator(base_generator, historical_data=historical_data)  # Pass instance, not class
forecast_insights = forecast_decorator.extract_insights(df)

# ABSA results
absa_decorator = CategoryABSAWithLLMInsightDecorator(base_generator)
absa_insights = absa_decorator.extract_insights(df)
print(absa_insights.head(5))

# Summarized insights
summary_decorator = CategorySummarizerDecorator(base_generator)
summary_insights = summary_decorator.extract_insights(df)
print(summary_insights.head(5))

# Combine insights
insights = absa_insights.merge(forecast_insights, on='category', how='outer').merge(summary_insights, on='category', how='outer')

# Extract insights
insights.to_csv("files/catgeory_analytics_2025.csv", index=False)