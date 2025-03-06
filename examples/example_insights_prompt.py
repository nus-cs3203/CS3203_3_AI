import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.llm_analysers.prompt_decorator import PromptGeneratorDecorator

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv").head(10)

# Apply decorator
base_generator = BaseInsightGenerator()
prompt_decorator = PromptGeneratorDecorator(base_generator)
insights = prompt_decorator.extract_insights(df)

print(insights)
