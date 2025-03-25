import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator
from insight_generator.category_analytics.llm_category_absa import CategoryABSAWithLLMInsightDecorator
from insight_generator.category_analytics.llm_category_summarizer import CategorySummarizerDecorator

# Sample Reddit posts
df = pd.read_csv("files/filtered_complaints.csv")
df.dropna(subset=["title"], inplace=True)
df = df.head(100)


# Apply decorators
base_generator = BaseInsightGenerator()

# Forecasted insights
forecast_decorator = TopicSentimentForecastDecorator(base_generator)  # Pass instance, not class
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
print(insights.head(5))
