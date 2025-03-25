import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
from insight_generator.base_decorator import InsightDecorator

def sentiment_to_label(sentiment):
    """Convert model sentiment output to label."""
    mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    return mapping.get(sentiment, "unknown")

class TopAdverseSentimentsDecoratorCAP(InsightDecorator):
    def __init__(self, wrapped, sentiment_col="title_with_desc_score",
                 category_col="category", text_col="description",
                 top_k=5, log_file="top_adverse_sentiments.txt"):
        """
        Identifies top adverse sentiments (5 most positive & 5 most negative per category)
        and applies Integrated Gradients for sentiment explanation.
        """
        super().__init__(wrapped)
        self.sentiment_col = sentiment_col
        self.category_col = category_col
        self.text_col = text_col
        self.top_k = top_k
        self.log_file = log_file
        
        # Load sentiment model
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        
        # Initialize Integrated Gradients
        self.ig = IntegratedGradients(self.model)
    
    def extract_insights(self, df):
        """Extracts top adverse sentiment posts per category and explains them."""
        insights = self._wrapped.extract_insights(df)
        
        if self.category_col not in df.columns or self.sentiment_col not in df.columns:
            raise ValueError(f"Required columns ('{self.category_col}', '{self.sentiment_col}') not found in df!")

        top_sentiments = {}

        # Open log file once for efficiency
        with open(self.log_file, "w", encoding="utf-8") as log_file:
            for category, group in df.groupby(self.category_col):
                # Get top 5 positive & negative sentiments
                top_positive = group.nlargest(self.top_k, self.sentiment_col)
                top_negative = group.nsmallest(self.top_k, self.sentiment_col)

                if top_positive.empty and top_negative.empty:
                    continue  # Skip empty categories

                # Combine and analyze explanations
                top_combined = pd.concat([top_positive, top_negative])
                explanations = self.explain_sentiments(top_combined[self.text_col].dropna().tolist())

                # Store insights
                top_sentiments[category] = {
                    "positive": top_positive[[self.text_col, self.sentiment_col]].to_dict(orient="records"),
                    "negative": top_negative[[self.text_col, self.sentiment_col]].to_dict(orient="records"),
                    "explanations": explanations
                }

                # Log results
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
        Uses Integrated Gradients to generate explanations for sentiment classification.
        """
        if not texts:
            return {}

        explanations = {}
        
        for text in texts:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                input_ids = inputs["input_ids"]
                
                # Compute Integrated Gradients
                attributions = self.ig.attribute(input_ids, target=0)  # Target sentiment class
                attributions = attributions.sum(dim=-1).squeeze(0).tolist()
                tokenized_text = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
                
                # Combine tokens with their attribution scores
                explanation = [(token, attr) for token, attr in zip(tokenized_text, attributions)]
                explanations[text] = explanation
            except Exception as e:
                explanations[text] = f"Error generating explanation: {str(e)}"

        return explanations
