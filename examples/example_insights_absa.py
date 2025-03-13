import pandas as pd

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator
from insight_generator.category_analytics.llm_category_absa import CategoryABSAWithLLMInsightDecorator
# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv").head(10)
df.dropna(subset=["title"], inplace=True)

# Apply decorator
base_generator = BaseInsightGenerator()
absa_decorator = CategoryABSAWithLLMInsightDecorator(base_generator)
insights = absa_decorator.extract_insights(df)

insights.to_csv("files/insights.csv", index=False)

print(insights)
