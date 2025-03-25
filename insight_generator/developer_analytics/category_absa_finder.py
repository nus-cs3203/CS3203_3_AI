import os
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator
from pyabsa import AspectTermExtraction as ATE

class CategoryABSAWithPyABSAInsightDecorator(InsightDecorator):
    """
    A decorator class that performs Aspect-Based Sentiment Analysis (ABSA) 
    on text data grouped by categories using PyABSA.

    Attributes:
        wrapped (InsightDecorator): The wrapped insight generator.
        text_col (str): The column containing text data for ABSA. If None, it will be inferred.
        category_col (str): The column containing category data. Default is "category".
    """
    def __init__(self, wrapped, text_col=None, category_col="category"):
        """
        Initializes the decorator with the wrapped insight generator and configuration.
        """
        super().__init__(wrapped)
        self.text_col = text_col
        self.category_col = category_col
        
        # Load PyABSA pre-trained model
        self.absa_model = ATE.AspectExtractor(model_name_or_path='multilingual')

    def extract_insights(self, df):
        """
        Extracts insights from the DataFrame by performing ABSA on text data grouped by categories.
        """
        insights = self._wrapped.extract_insights(df)

        if self.text_col is None:
            if {"title", "description"}.issubset(df.columns):
                df["title_with_description"] = df["title"].astype(str) + " " + df["description"].astype(str)
                self.text_col = "title_with_description"
            else:
                raise KeyError("Missing required text columns: title and description")

        absa_data = []
        for category, group in df.groupby(self.category_col):
            combined_text = " ".join(group[self.text_col].dropna().astype(str))
            if combined_text.strip():
                absa_result = self.perform_absa(combined_text)
                absa_data.append({
                    self.category_col: category,
                    "aspects": absa_result["aspects"],
                    "sentiments": absa_result["sentiments"]
                })
        
        absa_df = pd.DataFrame(absa_data)
        return df.merge(absa_df, on=self.category_col, how="left")

    def perform_absa(self, text):
        """
        Performs Aspect-Based Sentiment Analysis (ABSA) using PyABSA.
        """
        result = self.absa_model.extract_aspect(text)
        aspects = [aspect['aspect'] for aspect in result]
        sentiments = [aspect['sentiment'] for aspect in result]

        return {"aspects": aspects, "sentiments": sentiments}
