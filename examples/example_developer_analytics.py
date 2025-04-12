import pandas as pd
from insight_generator.base_insight_developer import BaseInsightDeveloperGenerator
from insight_generator.developer_analytics.cluster_maker import SentimentClusteringDecorator
from insight_generator.developer_analytics.sentiment_discrepancy_detector import SentimentDiscrepancyDecorator
from insight_generator.developer_analytics.anomaly_detector import SentimentAnomalyDetectionDecorator
from insight_generator.developer_analytics.importance_scorer import ImportanceScorerDecorator
from insight_generator.developer_analytics.sentiment_explainer_lime import TopAdverseSentimentsDecoratorLIME
from sentiment_analyser.context import SentimentAnalysisContext

# Load and preprocess data
df_base = pd.read_csv("final_processed_data_with_seconds.csv")
df_base['date'] = pd.to_datetime(df_base['date'], dayfirst=True, errors='coerce')
six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
historical_df = df_base[df_base['date'] < six_months_ago]
df = pd.read_csv("6_month_post_dev_analytics_sentiment.csv")


# Initialize base generator
base_generator = BaseInsightDeveloperGenerator()

# Explainer insights
explainer_decorator = TopAdverseSentimentsDecoratorLIME(base_generator)
explainer_insights = explainer_decorator.extract_insights(df)
print(explainer_insights.head(5))

# Cluster insights
cluster_detector = SentimentClusteringDecorator(base_generator)
cluster_insights = cluster_detector.extract_insights(df)
print(cluster_insights.head(5))

# Discrepancy insights
discrepancy_detector = SentimentDiscrepancyDecorator(base_generator, score_col_1="sentiment", score_col_2="comments_sentiment")
discrepancy_insights = discrepancy_detector.extract_insights(df)
print(discrepancy_insights.head(5))


# Anomaly detection insights
anomaly_detector = SentimentAnomalyDetectionDecorator(base_generator, sentiment_col="sentiment", historical_data=historical_df)
anomaly_insights = anomaly_detector.extract_insights(df)
print(anomaly_insights.head(5))

# Combine and display insights
insights = (explainer_insights
            .merge(cluster_insights, on='category', how='outer')
            .merge(discrepancy_insights, on='category', how='outer')
            .merge(anomaly_insights, on='category', how='outer'))
print(insights.head())
insights.to_csv("developer_insights.csv", index=False)
