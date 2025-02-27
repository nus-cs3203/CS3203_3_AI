from insight_generator.base_decorator import InsightDecorator
from keybert import KeyBERT
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

class TopicClusteringTrendDecorator(InsightDecorator):
    def __init__(self, wrapped_insight_generator, log_file="topic_trends.txt", 
                 text_col="selftext", category_col="Domain Category", time_col="created_utc", id_col=None):
        """
        Extracts salient words per category using KeyBERT.
        
        :param wrapped_insight_generator: Base insight generator
        :param log_file: File to log detected topics and trends
        :param text_col: Column containing text data
        :param category_col: Column containing category labels
        :param time_col: Column containing timestamps
        :param id_col: Column containing unique post IDs
        """
        super().__init__(wrapped_insight_generator)
        self.log_file = log_file
        self.text_col = text_col
        self.category_col = category_col
        self.time_col = time_col
        self.id_col = id_col
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.keyword_extractor = KeyBERT()

    def preprocess_text(self, text):
        """Removes stopwords, punctuation, and lemmatizes text."""
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return " ".join(words)

    def extract_insights(self, df):
        """
        Extracts salient keywords per category.
        :param df: DataFrame containing feedback
        :return: Insights with detected keywords per category
        """
        insights = self._wrapped.extract_insights(df)
        df[self.text_col] = df[self.text_col].apply(self.preprocess_text)
        df = df[df[self.text_col] != ""]  # Remove empty strings after preprocessing
        category_keywords = self.extract_keywords_per_category(df)
        self.log_keywords(category_keywords)
        insights["category_keywords"] = category_keywords
        return insights

    def extract_keywords_per_category(self, df):
        """
        Extracts keywords for each category by aggregating text within each category.
        :param df: DataFrame with feedback text and categories
        :return: Dictionary of category-wise keywords
        """
        category_keywords = {}
        for category, group in df.groupby(self.category_col):
            merged_text = " ".join(group[self.text_col].tolist())
            keywords = self.keyword_extractor.extract_keywords(merged_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
            category_keywords[category] = [kw[0] for kw in keywords]
        return category_keywords

    def log_keywords(self, category_keywords):
        """Logs detected keywords per category to a file."""
        with open(self.log_file, "a") as file:
            for category, keywords in category_keywords.items():
                file.write(f"Category: {category}\n")
                file.write(f"  Keywords: {', '.join(keywords)}\n")