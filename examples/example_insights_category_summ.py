import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.llm_category_summarizer import CategorySummarizerDecorator

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv").head(100)

# Apply decorator
base_generator = BaseInsightGenerator()
prompt_decorator = CategorySummarizerDecorator(base_generator)
insights = prompt_decorator.extract_insights(df)

insights.to_csv("files/category_summaries.csv", index=False)