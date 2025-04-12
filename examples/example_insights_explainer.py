import pandas as pd
from insight_generator.base_insight_developer import BaseInsightDeveloperGenerator
from insight_generator.developer_analytics.sentiment_explainer_lime import TopAdverseSentimentsDecoratorLIME

# Load and preprocess data
df = pd.read_csv("6_month_post_dev_analytics_sentiment.csv")


# Initialize base generator
base_generator = BaseInsightDeveloperGenerator()

# Forecasted insights
explainer_decorator = TopAdverseSentimentsDecoratorLIME(base_generator)
explainer_insights = explainer_decorator.extract_insights(df)
print(explainer_insights)