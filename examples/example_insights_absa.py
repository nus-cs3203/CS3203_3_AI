import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.llm_category_absa import CategoryABSAWithLLMInsightDecorator
from insight_generator.developer_analytics.category_absa_finder import CategoryABSAWithPyABSAInsightDecorator
from insight_generator.base_insight_developer import BaseInsightDeveloperGenerator

# Load and preprocess data
df = pd.read_csv("files/all_complaints_2022_2025.csv")
df.dropna(subset=["title"], inplace=True)

# Generate insights
base_generator = BaseInsightGenerator()
absa_decorator = CategoryABSAWithLLMInsightDecorator(base_generator)
insights = absa_decorator.extract_insights(df)
print(insights.head())


# Generate insights
base_generator = BaseInsightDeveloperGenerator()
absa_decorator = CategoryABSAWithPyABSAInsightDecorator(base_generator)
insights = absa_decorator.extract_insights(df)
print(insights.head())
