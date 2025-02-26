import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from insight_generator.base_decorator import InsightDecorator
from insight_generator.insight_interface import InsightGenerator

class SentimentClusteringDecorator(InsightDecorator):
    def __init__(self, wrapped: InsightGenerator, category_col="Domain Category", sentiment_cols=None, n_clusters=3):
        """
        Decorator for clustering sentiment scores within categories.

        :param wrapped: Wrapped InsightGenerator instance.
        :param category_col: Column containing category labels.
        :param sentiment_cols: List of sentiment score columns to use for clustering.
        :param n_clusters: Number of clusters per category.
        """
        super().__init__(wrapped)
        self.category_col = category_col
        self.sentiment_cols = sentiment_cols if sentiment_cols else ["sentiment_title_selftext_polarity", "sentiment_comments_polarity"]
        self.n_clusters = n_clusters

    def extract_insights(self, post, df):
        """
        Adds clustering-based insights to the extracted insights.

        :param post: Single Reddit post (row of DataFrame).
        :param df: Full DataFrame required for clustering calculations.
        :return: Modified insights dictionary.
        """
        insights = self._wrapped.extract_insights(post)
        category = post.get(self.category_col)

        if category is None or category not in df[self.category_col].unique():
            return insights  # Skip if category missing or invalid

        category_group = df[df[self.category_col] == category]
        if len(category_group) < self.n_clusters:
            return insights  # Skip clustering if not enough data points
        
        X = category_group[self.sentiment_cols].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        category_group = category_group.copy()
        category_group["cluster"] = labels

        # Assign cluster label to the post if it exists in the clustered group
        cluster_label = category_group.loc[category_group.index == post.name, "cluster"]
        insights["sentiment_cluster"] = int(cluster_label.iloc[0]) if not cluster_label.empty else None

        self._log_cluster(post, category, insights["sentiment_cluster"])

        return insights

    def _log_cluster(self, post, category, cluster_label):
        """Logs sentiment clustering results into a text file."""
        log_entry = (f"Post ID: {post.get('id', 'N/A')} | "
                     f"Category: {category} | "
                     f"Assigned Cluster: {cluster_label}\n")
        with open("sentiment_clusters_log.txt", "a") as log_file:
            log_file.write(log_entry)
