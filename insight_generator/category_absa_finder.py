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
        Logs ABSA results into a file.
        
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
        
        # Log ABSA results
        self._log_absa_results(absa_results)
        
        insights["absa_results"] = absa_results
        return insights
    
    def _perform_absa(self, texts):
        """Perform Aspect-Based Sentiment Analysis (ABSA) on a batch of texts."""
        sentiments = self.absa_pipeline(texts)
        return {text: sentiment["label"] for text, sentiment in zip(texts, sentiments)}
    
    def _log_absa_results(self, absa_results):
        """Logs ABSA results into a text file."""
        with open("absa_results_log.txt", "a") as log_file:
            for category, results in absa_results.items():
                log_file.write(f"Category: {category}\n")
                for text, sentiment in results.items():
                    log_file.write(f"- {text[:100]}... Sentiment: {sentiment}\n")
                log_file.write("\n")
