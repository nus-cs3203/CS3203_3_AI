import spacy
from insight_generator.base_decorator import InsightDecorator
import pytextrank

class KeywordsTrendDecorator(InsightDecorator):
    def __init__(self, wrapped_insight_generator, log_file="topic_trends.txt", 
                 text_col="title", category_col="category", time_col="created_utc", id_col=None):
        """
        Extracts salient words per category.
        
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
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textrank")

    def extract_insights(self, df):
        """
        Extracts salient keywords per category.
        :param df: DataFrame containing feedback
        :return: Insights with detected keywords per category
        """
        insights = self._wrapped.extract_insights(df)
        df = df[df[self.text_col].notna() & (df[self.text_col] != "")]
        category_keywords = self.extract_keywords_per_category(df)
        insights["keywords"] = category_keywords
        return insights

    def extract_keywords_per_category(self, df):
        """
        Extracts keywords for each category by aggregating text within each category.
        :param df: DataFrame with feedback text and categories
        :return: Dictionary of category-wise keywords
        """
        category_keywords = {}
        for category, group in df.groupby(self.category_col):
            merged_text = " ".join(group[self.text_col])
            merged_text = " ".join([word for word in merged_text.split() if "Singapore" not in word])
            doc = self.nlp(merged_text)
            seen_keywords = set()
            keywords = []
            for phrase in doc._.phrases:
                if len(keywords) >= 5:
                    break
                if phrase.text not in seen_keywords:
                    keywords.append(phrase.text)
                    seen_keywords.add(phrase.text)
            category_keywords[category] = keywords
        return category_keywords

