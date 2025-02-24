import pandas as pd
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
        """Cluster posts based on selftext using KMeans"""
        tfidf_matrix = self.vectorizer.fit_transform(posts_df['selftext'])
        
        # Apply KMeans clustering
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        posts_df['cluster'] = self.kmeans.fit_predict(tfidf_matrix)

    def extract_insights(self, posts_df):
        """Override the base method to include clustering insights for batch processing"""
        # Apply the base method for each post and cluster them
        insights_list = []
        
        if self.kmeans is None:  # Perform clustering if not already done
            self.cluster_posts(posts_df)
        
        for _, row in posts_df.iterrows():
            insights = super().extract_insights(row)  # Get existing insights from the wrapped component
            
            # Assign the post to its cluster based on the 'selftext'
            tfidf_matrix = self.vectorizer.transform([row['selftext']])
            cluster = self.kmeans.predict(tfidf_matrix)[0]  # Assign to a cluster
            
            insights["cluster"] = cluster
            insights_list.append(insights)
        
        return pd.DataFrame(insights_list)
