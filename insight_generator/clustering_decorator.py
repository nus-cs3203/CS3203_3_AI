import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from insight_generator.base_decorator import InsightDecorator

class ClusteringDecorator(InsightDecorator):
    def __init__(
        self, wrapped_component, 
        num_clusters=3, 
        column_mappings=None
    ):
        super().__init__(wrapped_component)
        self.num_clusters = num_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kmeans = None
        self.report_data = []

        self.column_mappings = column_mappings if column_mappings else {
            "text": "selftext"
        }

    def cluster_posts(self, posts_df):
        """Cluster posts based on text content using KMeans."""
        text_column = self.column_mappings["text"]

        if text_column not in posts_df:
            raise KeyError(f"Column '{text_column}' not found in DataFrame.")

        posts_df[text_column] = posts_df[text_column].fillna("")
        tfidf_matrix = self.vectorizer.fit_transform(posts_df[text_column])

        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        posts_df["cluster"] = self.kmeans.fit_predict(tfidf_matrix)

        for _, row in posts_df.iterrows():
            cluster = row["cluster"]
            summary = row.get("summary", "No summary available")
            self.report_data.append({"Cluster": cluster, "Summary": summary})

    def extract_insights(self, posts_df):
        """Override the base method to include clustering insights."""
        insights_list = []

        if self.kmeans is None:
            self.cluster_posts(posts_df)

        text_column = self.column_mappings["text"]

        for _, row in posts_df.iterrows():
            insights = super().extract_insights(row)
            tfidf_matrix = self.vectorizer.transform([row.get(text_column, "")])
            cluster = self.kmeans.predict(tfidf_matrix)[0]

            insights["cluster"] = cluster
            insights_list.append(insights)

        return pd.DataFrame(insights_list)

    def generate_report(self, report_path="cluster_report.txt"):
        cluster_grouped = {}

        for entry in self.report_data:
            cluster = entry["Cluster"]

            if cluster not in cluster_grouped:
                cluster_grouped[cluster] = []
            cluster_grouped[cluster].append(entry)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Cluster-wise Summary Report ===\n\n")
            for cluster, summaries in cluster_grouped.items():
                f.write(f"Cluster: {cluster}\n")
                f.write(f"Total Entries: {len(summaries)}\n")
                f.write("Sample Summaries:\n")
                for sample in summaries[:3]:
                    f.write(f"- {sample['Summary']}\n")
                f.write("\n" + "-"*50 + "\n\n")

        print(f"Cluster-wise summary report saved to {report_path}")
