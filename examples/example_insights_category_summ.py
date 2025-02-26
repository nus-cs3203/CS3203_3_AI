import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.llm_category_summarizer import CategorySummarizerDecorator

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv").head(10)

# Apply decorator
base_generator = BaseInsightGenerator()
prompt_decorator = CategorySummarizerDecorator(base_generator)
insights = prompt_decorator.extract_insights(df)

print(insights)
