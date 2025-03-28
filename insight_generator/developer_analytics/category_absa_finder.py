import os
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator
from pyabsa import AspectTermExtraction as ATEPC

class CategoryABSAWithPyABSAInsightDecorator(InsightDecorator):
    """
    A decorator class that performs Aspect-Based Sentiment Analysis (ABSA) 
    on text data grouped by categories using PyABSA.
    """
    def __init__(self, wrapped, text_col=None, category_col="category"):
        super().__init__(wrapped)
        self.text_col = text_col
        self.category_col = category_col
        self.absa_model = ATEPC.AspectExtractor("multilingual")

    def extract_insights(self, df):
        insights = self._wrapped.extract_insights(df)

        if self.text_col is None:
            if {"title", "description"}.issubset(df.columns):
                df["text_combined"] = df["title"].astype(str) + " " + df["description"].astype(str)
                self.text_col = "text_combined"
            else:
                raise KeyError("Missing required text columns: title and description")
        
        absa_results = []
        for category, group in df.groupby(self.category_col):
            combined_text = " ".join(group[self.text_col].dropna().astype(str))
            if combined_text.strip():
                absa_result = self.perform_absa(combined_text)
                absa_results.append({
                    self.category_col: category,
                    "aspects": absa_result["aspects"],
                    "sentiments": absa_result["sentiments"]
                })
        
        absa_df = pd.DataFrame(absa_results)
        return absa_df

    def perform_absa(self, text):
        result = self.absa_model.predict(text)
        return {"aspects": result.get('aspect', []), "sentiments": result.get('sentiment', [])}
