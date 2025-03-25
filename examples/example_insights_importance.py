import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.developer_analytics.importance_scorer import ImportanceScorerDecorator

# Load data
df = pd.read_csv("files/all_complaints_2022_2025.csv")

# Generate insights with importance scoring
base_generator = BaseInsightGenerator()
imp_decorator = ImportanceScorerDecorator(base_generator)
insights = imp_decorator.extract_insights(df)

print(insights)
