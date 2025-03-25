import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
from insight_generator.base_decorator import InsightDecorator

class TopAdverseSentimentsDecoratorLIME(InsightDecorator):
    def __init__(self, wrapped, sentiment_col="title_with_desc_score",
                 category_col="category", text_col="description",
                 top_k=5, log_file="top_adverse_sentiments.txt"):
        """
        Decorator to identify top adverse sentiments (most positive and most negative posts) 
        per category and explain them using LIME.

        Args:
        - wrapped: The base insight generator to decorate.
        - sentiment_col: Column containing sentiment polarity scores.
        - category_col: Column indicating the topic/category of the posts.
        - text_col: Column containing the text content of the posts.
        - top_k: Number of extreme positive and negative samples to extract per category.
        - log_file: File path to log the insights.
        """
        super().__init__(wrapped)
        self.sentiment_col = sentiment_col
        self.category_col = category_col
        self.text_col = text_col
        self.top_k = top_k
        self.log_file = log_file
        
        # Load sentiment analysis model
        self.sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

        # Initialize LIME text explainer
        self.explainer = LimeTextExplainer(class_names=["negative", "neutral", "positive"])
    
    def extract_insights(self, df):
        """
        Extracts insights by identifying top adverse sentiment posts per category 
        and generating explanations for them using LIME.

        Args:
        - df: DataFrame containing the data to analyze.

        Returns:
        - A dictionary containing the original insights and the top sentiments with explanations.
        """
        insights = self._wrapped.extract_insights(df)
        
        if self.category_col not in df.columns or self.sentiment_col not in df.columns:
            raise ValueError(f"Required columns ('{self.category_col}', '{self.sentiment_col}') not found in df!")

        top_sentiments = {}

        # Open log file for writing insights
        with open(self.log_file, "w", encoding="utf-8") as log_file:
            for category, group in df.groupby(self.category_col):
                # Get top k positive and negative sentiment posts
                top_positive = group.nlargest(self.top_k, self.sentiment_col)
                top_negative = group.nsmallest(self.top_k, self.sentiment_col)

                if top_positive.empty and top_negative.empty:
                    continue  # Skip categories with no data

                # Combine positive and negative posts for explanation
                top_combined = pd.concat([top_positive, top_negative])
                explanations = self.explain_sentiments(top_combined[self.text_col].dropna().tolist())

                # Store insights for the category
                top_sentiments[category] = {
                    "positive": top_positive[[self.text_col, self.sentiment_col]].to_dict(orient="records"),
                    "negative": top_negative[[self.text_col, self.sentiment_col]].to_dict(orient="records"),
                    "explanations": explanations
                }

                # Log insights to the file
                log_file.write(f"\nCategory: {category}\n")
                log_file.write(f"Top {self.top_k} Positive:\n")
                for _, row in top_positive.iterrows():
                    log_file.write(f"- Score: {row[self.sentiment_col]:.4f}, Text: {row[self.text_col][:100]}...\n")
                
                log_file.write(f"\nTop {self.top_k} Negative:\n")
                for _, row in top_negative.iterrows():
                    log_file.write(f"- Score: {row[self.sentiment_col]:.4f}, Text: {row[self.text_col][:100]}...\n")
                
                log_file.write("\nExplanations:\n")
                for text, explanation in explanations.items():
                    log_file.write(f"- {text[:100]}...: {explanation}\n")

        insights["top_sentiments"] = top_sentiments
        return insights

    def explain_sentiments(self, texts):
        """
        Generates explanations for sentiment classification using LIME.

        Args:
        - texts: List of text samples to explain.

        Returns:
        - A dictionary mapping each text to its LIME explanation or an error message.
        """
        if not texts:
            return {}

        explanations = {}

        def predict_proba(text_samples):
            """
            Predicts probability distributions for sentiment classes 
            to be used by LIME for explanation.

            Args:
            - text_samples: List of text samples.

            Returns:
            - A NumPy array of probability distributions for each sample.
            """
            outputs = self.sentiment_model(text_samples)
            proba_list = []

            for output in outputs:
                label = output["label"]
                score = output["score"]
                if label == "LABEL_0":
                    proba_list.append([score, 0, 0])  # Negative
                elif label == "LABEL_1":
                    proba_list.append([0, score, 0])  # Neutral
                elif label == "LABEL_2":
                    proba_list.append([0, 0, score])  # Positive
                else:
                    proba_list.append([0.33, 0.33, 0.33])  # Fallback to uniform distribution

            return np.array(proba_list)

        for text in texts:
            try:
                # Generate LIME explanation for the text
                exp = self.explainer.explain_instance(text, predict_proba, num_features=10)
                explanations[text] = exp.as_list()  # Convert explanation to a readable format
            except Exception as e:
                explanations[text] = f"Error generating explanation: {str(e)}"

        return explanations
