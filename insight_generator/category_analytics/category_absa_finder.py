import pandas as pd
from insight_generator.base_decorator import InsightDecorator
from insight_generator.insight_interface import InsightGenerator
from collections import defaultdict
from transformers import pipeline

class CategoryABSAInsightDecorator(InsightDecorator):
    def __init__(self, wrapped: InsightGenerator, 
                 text_col="title_selftext", 
                 category_col="Domain Category"):
        """
        Decorator to perform Aspect-Based Sentiment Analysis (ABSA) per category.
        
        :param wrapped: The wrapped InsightGenerator instance.
        :param text_col: Column containing text to analyze.
        :param category_col: Column containing categories.
        """
        super().__init__(wrapped)
        self.text_col = text_col
        self.category_col = category_col
        self.absa_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    
    def extract_insights(self, post_df):
        insights = self._wrapped.extract_insights(post_df)
        
        # Group text by category
        category_texts = defaultdict(list)
        for _, row in post_df.iterrows():
            category = row.get(self.category_col, "Unknown")
            text = row.get(self.text_col, "")
            if text:
                category_texts[category].append(text)
        
        # Perform ABSA analysis
        absa_results = {}
        for category, texts in category_texts.items():
            absa_results[category] = self._perform_absa(texts)
        
        # Format ABSA results
        formatted_absa_results = self._format_absa_results(absa_results)
        
        insights["absa_results"] = formatted_absa_results
        return insights
    
    def _perform_absa(self, texts):
        """Perform Aspect-Based Sentiment Analysis (ABSA) on a batch of texts."""
        sentiments = self.absa_pipeline(texts)
        return {text: sentiment["label"] for text, sentiment in zip(texts, sentiments)}
    
    def _format_absa_results(self, absa_results):
        """Format ABSA results into the form of 'THEME, sentiment'."""
        formatted_results = {}
        for category, results in absa_results.items():
            formatted_results[category] = [f"{text[:100]}, {sentiment}" for text, sentiment in results.items()]
        return formatted_results
