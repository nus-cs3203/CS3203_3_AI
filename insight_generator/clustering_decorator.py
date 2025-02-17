from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from insight_generator.base_decorator import InsightDecorator

class ClusteringDecorator(InsightDecorator):
    def __init__(self, wrapped_component, num_clusters=3):
        super().__init__(wrapped_component)
        self.num_clusters = num_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kmeans = None

    def cluster_posts(self, posts_df):
        """Cluster posts based on sentiment or text using KMeans"""
        # Use the selftext for clustering
        tfidf_matrix = self.vectorizer.fit_transform(posts_df['selftext'])
        
        # Apply KMeans clustering
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        posts_df['cluster'] = self.kmeans.fit_predict(tfidf_matrix)

    def extract_insights(self, post):
        """Override the base method to include clustering insights"""
        insights = super().extract_insights(post)  # Get existing insights from the wrapped component
        
        # Here you would assign the post to a cluster based on its 'selftext'
        tfidf_matrix = self.vectorizer.transform([post['selftext']])
        cluster = self.kmeans.predict(tfidf_matrix)[0]  # Assign to a cluster
        
        insights["cluster"] = cluster
        return insights
