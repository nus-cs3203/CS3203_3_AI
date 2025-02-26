from insight_generator.base_decorator import InsightDecorator
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

class TopicClusteringTrendDecorator(InsightDecorator):
    def __init__(self, wrapped_insight_generator, log_file="topic_trends.txt", 
                 text_col="selftext", time_col="created_utc", use_bertopic=True):
        """
        Implements unsupervised clustering (BERTopic or LDA) to group feedback and detect trends.
        
        :param wrapped_insight_generator: Base insight generator
        :param log_file: File to log detected topics and trends
        :param text_col: Column containing text data
        :param time_col: Column containing timestamps
        :param use_bertopic: Whether to use BERTopic (True) or LDA (False)
        """
        super().__init__(wrapped_insight_generator)
        self.log_file = log_file
        self.text_col = text_col
        self.time_col = time_col
        self.use_bertopic = use_bertopic

    def extract_insights(self, df):
        """
        Extracts topic clusters and trends from feedback data.
        :param df: DataFrame containing feedback
        :return: Insights with detected topics and trends
        """
        insights = self._wrapped.extract_insights(df)
        topics = self.detect_topics(df)
        self.log_trends(topics)
        insights["topics"] = topics
        return insights

    def detect_topics(self, df):
        """
        Clusters feedback into topics using BERTopic or LDA.
        :param df: DataFrame with feedback text
        :return: Dictionary of detected topics
        """
        df = df.dropna(subset=[self.text_col])
        texts = df[self.text_col].tolist()

        if self.use_bertopic:
            model = BERTopic()
            topics, _ = model.fit_transform(texts)
        else:
            vectorizer = CountVectorizer(stop_words="english", max_features=5000)
            text_matrix = vectorizer.fit_transform(texts)
            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            topics = lda.fit_transform(text_matrix).argmax(axis=1)

        df["topic"] = topics
        return df["topic"].value_counts().to_dict()  # Topic distribution

    def log_trends(self, topics):
        """Logs detected topics to a file."""
        with open(self.log_file, "a") as file:
            file.write(f"Detected Topics: {topics}\n")
