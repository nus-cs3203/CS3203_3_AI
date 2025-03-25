import pandas as pd
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.developer_analytics.anomaly_detector import SentimentAnomalyDetectionDecorator

# Load historical sentiment data
historical_data = pd.read_csv("files/all_complaints_2022_2025.csv")

# Load Reddit posts
df = pd.read_csv("files/all_complaints_2022_2025.csv")

# Apply anomaly detector
base_generator = BaseInsightGenerator()
anomaly_detector = SentimentAnomalyDetectionDecorator(base_generator, historical_data)

# Extract and print insights
insights = anomaly_detector.extract_insights(df)
print(insights["dates_with_shift"])
