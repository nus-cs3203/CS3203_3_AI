import pandas as pd
from insight_generator.base_insight_developer import BaseInsightDeveloperGenerator
from insight_generator.developer_analytics.anomaly_detector import SentimentAnomalyDetectionDecorator

# Load historical sentiment data
historical_data = pd.read_csv("files/all_complaints_2022_2025.csv")
historical_data.dropna(subset=["title"], inplace=True)
# Load Reddit posts
df = pd.read_csv("files/all_complaints_2022_2025.csv")
df.dropna(subset=["title"], inplace=True)

# Apply anomaly detector
base_generator = BaseInsightDeveloperGenerator()
anomaly_detector = SentimentAnomalyDetectionDecorator(base_generator, historical_data)

# Extract and print insights
insights = anomaly_detector.extract_insights(df)
print(insights["dates_with_shift"])
