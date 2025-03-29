import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from insight_generator.base_decorator import InsightDecorator
from insight_generator.insight_interface import InsightGenerator

class SentimentClusteringDecorator(InsightDecorator):
    def __init__(self, wrapped: InsightGenerator, 
                 category_col="category", 
                 sentiment_cols=["sentiment"], 
                 n_clusters=10, 
                 log_file="sentiment_clusters.txt"):
        """
        A decorator for clustering sentiment scores within categories.

        :param wrapped: The wrapped InsightGenerator instance.
        :param category_col: The column containing category labels.
        :param sentiment_cols: List of sentiment score columns to use for clustering. Defaults to 
                               ["title_with_desc_score", "comments_score"] if not provided.
        :param n_clusters: The number of clusters to create per category.
        :param log_file: The file path to log clustering results.
        """
        super().__init__(wrapped)
        self.category_col = category_col
        self.sentiment_cols = sentiment_cols if sentiment_cols else [
            "title_with_desc_score", 
            "comments_score"
        ]
        self.n_clusters = n_clusters
        self.log_file = log_file

    def extract_insights(self, df: pd.DataFrame):
        """
        Extracts insights, including clustering sentiment scores within categories.

        :param df: A pandas DataFrame containing sentiment data.
        :return: A pandas DataFrame with categories and their corresponding clusters.
        """
        # Extract base insights from the wrapped generator
        self._wrapped.extract_insights(df)
        category_clusters = []

        with open(self.log_file, "w") as log_file:
            # Group data by category and perform clustering for each group
            for category, category_group in df.groupby(self.category_col):
                # Drop rows with missing sentiment scores
                category_group = category_group.dropna(subset=self.sentiment_cols)
                
                # Skip clustering if there is insufficient data for the specified number of clusters
                if len(category_group) < self.n_clusters:
                    continue

                # Prepare data for clustering
                X = category_group[self.sentiment_cols].values
                X_scaled = StandardScaler().fit_transform(X)  # Standardize features

                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)

                # Add cluster labels to the DataFrame
                category_group = category_group.copy()
                category_group["cluster"] = labels

                # Organize post IDs by cluster
                clusters = [[] for _ in range(self.n_clusters)]
                for idx, row in category_group.iterrows():
                    post_id = str(idx)  # Use the index as the post ID
                    cluster_label = int(row["cluster"])
                    clusters[cluster_label].append(post_id)

                # Append category and clusters to the result
                category_clusters.append({"category": category, "clusters": clusters})

                # Write clustering results to the log file
                log_file.write(f"Category: {category}\n")
                for cluster_label, post_ids in enumerate(clusters):
                    log_file.write(f"  Cluster {cluster_label}: {', '.join(post_ids)}\n")
                log_file.write("\n")

        # Convert the result to a DataFrame
        result_df = pd.DataFrame(category_clusters)
        return result_df
