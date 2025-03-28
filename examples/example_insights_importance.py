import pandas as pd
from insight_generator.base_insight_developer import BaseInsightDeveloperGenerator
from insight_generator.developer_analytics.importance_scorer import ImportanceScorerDecorator

# Load data
df = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Generate insights with importance scoring
base_generator = BaseInsightDeveloperGenerator()
imp_decorator = ImportanceScorerDecorator(base_generator, sentiment_col_1="sentiment_title_selftext_polarity", sentiment_col_2="sentiment_comments_polarity")
insights = imp_decorator.extract_insights(df)

print(insights)
