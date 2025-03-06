import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.post_analytics.importance_scorer import ImportanceScorerDecorator

# Sample Reddit posts
df = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Apply decorator
base_generator = BaseInsightGenerator()
imp_decorator = ImportanceScorerDecorator(base_generator)
insights = imp_decorator.extract_insights(df)

print(insights)
