import pandas as pd

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_summary_decorator import CategorySummarizerDecorator
from insight_generator.clustering_decorator import SentimentClustering
from insight_generator.sentiment_discrepancy import SentimentDiscrepancyDecorator

# Step 1: Read CSV into DataFrame
csv_file = "files/sentiment_scored_2023_data.csv"  # Change to your actual file path
df = pd.read_csv(csv_file)
df['title_selftext'] = df['title'] + " " + df['selftext']

# Step 2: Initialize the base insight generator
base_generator = BaseInsightGenerator()

# Step 3: Apply Sentiment Discrepancy Decorator
discrepancy_decorator = SentimentDiscrepancyDecorator(base_generator)
df["insights"] = df.apply(lambda row: discrepancy_decorator.extract_insights(row), axis=1)

# Step 4: Apply Category Summarizer Decorator
summarizer_decorator = CategorySummarizerDecorator(base_generator)
df["summaries"] = df.apply(lambda row: summarizer_decorator.extract_insights(row), axis=1)

# Step 5: Apply Sentiment Clustering
clustering = SentimentClustering()
clusters = clustering.cluster_sentiments(df)

# Step 6: Save updated DataFrame with insights
df.to_csv("processed_reddit_posts.csv", index=False)

print("Processing complete! Insights and logs are saved.")
