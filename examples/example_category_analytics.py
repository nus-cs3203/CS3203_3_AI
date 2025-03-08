import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator
from insight_generator.category_analytics.llm_category_absa import CategoryABSAWithLLMInsightDecorator
from insight_generator.category_analytics.llm_category_summarizer import CategorySummarizerDecorator

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv").tail(15)
df.dropna(subset=["title"], inplace=True)

# Sample historical sentiment data
historical_data = pd.read_csv("files/sentiment_scored_2023_data.csv").head(15)

# Apply decorators
base_generator = BaseInsightGenerator()

# Forecasted insights
forecast_decorator = TopicSentimentForecastDecorator(base_generator, historical_data)  # Pass instance, not class
forecast_insights = forecast_decorator.extract_insights(df)
print(forecast_insights.head(5))
forecast_insights.rename(columns={'category': 'Category'}, inplace=True)

# ABSA results
absa_decorator = CategoryABSAWithLLMInsightDecorator(base_generator)
absa_insights = absa_decorator.extract_insights(df)
print(absa_insights.head(5))
absa_insights.rename(columns={'Domain Category': 'Category'}, inplace=True)

# Summarized insights
summary_decorator = CategorySummarizerDecorator(base_generator)
summary_insights = summary_decorator.extract_insights(df)
print(summary_insights.head(5))
summary_insights.rename(columns={'Domain Category': 'Category'}, inplace=True)

# Combine insights
insights = absa_insights.merge(forecast_insights, on='Category', how='outer').merge(summary_insights, on='Category', how='outer')

# Extract insights
print(insights.head(5))
