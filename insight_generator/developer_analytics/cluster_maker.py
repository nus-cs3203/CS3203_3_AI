import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from insight_generator.base_decorator import InsightDecorator
from insight_generator.insight_interface import InsightGenerator

class SentimentClusteringDecorator(InsightDecorator):
    def __init__(self, wrapped: InsightGenerator, 
                 category_col="category", 
                 sentiment_cols=None, 
                 n_clusters=3, 
                 log_file="sentiment_clusters.txt"):
        """
        Decorator for clustering sentiment scores within categories.

        :param wrapped: Wrapped InsightGenerator instance.
        :param category_col: Column containing category labels.
        :param sentiment_cols: List of sentiment score columns to use for clustering.
        :param n_clusters: Number of clusters per category.
        :param log_file: File to log clustering results.
        """
        super().__init__(wrapped)
        self.category_col = category_col
        self.sentiment_cols = sentiment_cols if sentiment_cols else [
            "sentiment_title_selftext_polarity", 
            "sentiment_comments_polarity"
        ]
        self.n_clusters = n_clusters
        self.log_file = log_file

    def extract_insights(self, df: pd.DataFrame):
        """
        Extracts insights including sentiment clustering.

        :param df: DataFrame containing sentiment data.
        :return: Insights dictionary with sentiment clusters.
        """
        insights = self._wrapped.extract_insights(df)
        category_clusters = defaultdict(lambda: defaultdict(list))

        with open(self.log_file, "w") as log_file:
            for category, category_group in df.groupby(self.category_col):
                category_group = category_group.dropna(subset=self.sentiment_cols)
                if len(category_group) < self.n_clusters:
                    continue  # Skip clustering if insufficient data

                # Perform clustering
                X = category_group[self.sentiment_cols].values
                X_scaled = StandardScaler().fit_transform(X)
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)

                # Assign cluster labels
                category_group = category_group.copy()
                category_group["cluster"] = labels

                for idx, row in category_group.iterrows():
                    post_id = str(idx)
                    cluster_label = int(row["cluster"])
                    category_clusters[category][cluster_label].append(post_id)

            # Write structured log output
            for category, clusters in category_clusters.items():
                log_file.write(f"Category: {category}\n")
                for cluster_label, post_ids in clusters.items():
                    log_file.write(f"  Cluster {cluster_label}: {', '.join(post_ids)}\n")
                log_file.write("\n")

        insights["sentiment_clusters"] = category_clusters
        return insights
