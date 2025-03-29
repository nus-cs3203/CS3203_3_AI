import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
from insight_generator.base_decorator import InsightDecorator

def sentiment_to_label(sentiment):
    """
    Convert the sentiment output from the model to a human-readable label.

    Args:
        sentiment (str): Sentiment label from the model (e.g., "LABEL_0").

    Returns:
        str: Human-readable sentiment label ("negative", "neutral", "positive", or "unknown").
    """
    mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    return mapping.get(sentiment, "unknown")

class TopAdverseSentimentsDecoratorCAP(InsightDecorator):
    def __init__(self, wrapped, sentiment_col="title_with_desc_score",
                 category_col="category", text_col="description",
                 top_k=5, log_file="top_adverse_sentiments.txt"):
        """
        Decorator to identify and explain top adverse sentiments in a dataset.

        Args:
            wrapped (InsightDecorator): The wrapped insight generator.
            sentiment_col (str): Column name containing sentiment scores.
            category_col (str): Column name containing category labels.
            text_col (str): Column name containing text data.
            top_k (int): Number of top positive and negative sentiments to extract per category.
            log_file (str): File path to log the results.
        """
        super().__init__(wrapped)
        self.sentiment_col = sentiment_col
        self.category_col = category_col
        self.text_col = text_col
        self.top_k = top_k
        self.log_file = log_file
        
        # Load sentiment analysis model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        
        # Initialize Integrated Gradients for explanation
        self.ig = IntegratedGradients(self.model)
    
    def extract_insights(self, df):
        """
        Extract top adverse sentiment posts per category and generate explanations.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data.

        Returns:
            dict: Insights including top positive/negative sentiments and their explanations.
        """
        insights = self._wrapped.extract_insights(df)
        
        if self.category_col not in df.columns or self.sentiment_col not in df.columns:
            raise ValueError(f"Required columns ('{self.category_col}', '{self.sentiment_col}') not found in df!")

        top_sentiments = {}

        # Open log file for writing results
        with open(self.log_file, "w", encoding="utf-8") as log_file:
            for category, group in df.groupby(self.category_col):
                # Get top k positive and negative sentiments
                top_positive = group.nlargest(self.top_k, self.sentiment_col)
                top_negative = group.nsmallest(self.top_k, self.sentiment_col)

                if top_positive.empty and top_negative.empty:
                    continue  # Skip categories with no data

                # Combine positive and negative sentiments for explanation
                top_combined = pd.concat([top_positive, top_negative])
                explanations = self.explain_sentiments(top_combined[self.text_col].dropna().tolist())

                # Store insights for the category
                top_sentiments[category] = {
                    "positive": top_positive[[self.text_col, self.sentiment_col]].to_dict(orient="records"),
                    "negative": top_negative[[self.text_col, self.sentiment_col]].to_dict(orient="records"),
                    "explanations": explanations
                }

                # Log results to the file
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
        Generate explanations for sentiment classification using Integrated Gradients.

        Args:
            texts (list of str): List of text inputs to explain.

        Returns:
            dict: Explanations for each text, mapping text to token-attribution pairs.
        """
        if not texts:
            return {}

        explanations = {}
        
        for text in texts:
            try:
                # Tokenize the input text
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                input_ids = inputs["input_ids"]
                
                # Compute attributions using Integrated Gradients
                attributions = self.ig.attribute(input_ids, target=0)  # Target sentiment class
                attributions = attributions.sum(dim=-1).squeeze(0).tolist()
                tokenized_text = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
                
                # Combine tokens with their attribution scores
                explanation = [(token, attr) for token, attr in zip(tokenized_text, attributions)]
                explanations[text] = explanation
            except Exception as e:
                explanations[text] = f"Error generating explanation: {str(e)}"

        return explanations
