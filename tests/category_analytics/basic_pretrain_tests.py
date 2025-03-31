import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator
from insight_generator.category_analytics.llm_category_absa import CategoryABSAWithLLMInsightDecorator
from insight_generator.category_analytics.llm_category_summarizer import CategorySummarizerDecorator

# Load and preprocess data
df = pd.read_csv("files/all_complaints_2022_2025.csv").head(1000)
df.dropna(subset=["title"], inplace=True)

# Initialize base generator
base_generator = BaseInsightGenerator()

# Forecasted insights
forecast_decorator = TopicSentimentForecastDecorator(base_generator)
forecast_insights = forecast_decorator.extract_insights(df)

# Validate forecasted insights
assert isinstance(forecast_insights, pd.DataFrame), "Forecasted insights should be a DataFrame"
assert not forecast_insights.isnull().values.any(), "Forecasted insights contain missing values"
expected_forecast_cols = {"category", "forecasted_sentiment"}
missing_cols = expected_forecast_cols - set(forecast_insights.columns)
assert not missing_cols, f"Missing columns in forecasted insights: {missing_cols}"

# ABSA insights
absa_decorator = CategoryABSAWithLLMInsightDecorator(base_generator)
absa_insights = absa_decorator.extract_insights(df)

# Validate ABSA insights
assert isinstance(absa_insights, pd.DataFrame), "ABSA insights should be a DataFrame"
assert not absa_insights.isnull().values.any(), "ABSA insights contain missing values"
expected_absa_cols = {"category", "absa_result", "keywords"}
missing_absa_cols = expected_absa_cols - set(absa_insights.columns)
assert not missing_absa_cols, f"Missing columns in ABSA insights: {missing_absa_cols}"

# Summarized insights
summary_decorator = CategorySummarizerDecorator(base_generator)
summary_insights = summary_decorator.extract_insights(df)

# Validate summarized insights
assert isinstance(summary_insights, pd.DataFrame), "Summarized insights should be a DataFrame"
assert not summary_insights.isnull().values.any(), "Summarized insights contain missing values"
expected_summary_cols = {"category", "summary", "concerns", "suggestions"}
missing_summary_cols = expected_summary_cols - set(summary_insights.columns)
assert not missing_summary_cols, f"Missing columns in summary insights: {missing_summary_cols}"

# Combine and validate merged insights
insights = absa_insights.merge(forecast_insights, on='category', how='outer').merge(summary_insights, on='category', how='outer')

# Validate merged insights
assert isinstance(insights, pd.DataFrame), "Final insights should be a DataFrame"
assert not insights.isnull().values.any(), "Final merged insights contain missing values"
expected_final_cols = {"category", "absa_result", "keywords", "forecasted_sentiment", "summary", "concerns", "suggestions", "sentiment"}
missing_final_cols = expected_final_cols - set(insights.columns)
assert not missing_final_cols, f"Missing expected columns in final insights: {missing_final_cols}"

# Validate the shape of the final insights DataFrame
unique_categories = df["category"].nunique()
expected_shape = (unique_categories, len(expected_final_cols)) 

print(insights.head(5))
assert insights.shape == expected_shape, f"Final insights shape mismatch: expected {expected_shape}, got {insights.shape}"
print("All assertions passed! Pipeline is functioning correctly.")
