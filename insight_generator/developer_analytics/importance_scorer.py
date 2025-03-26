import os
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator

class ImportanceScorerDecorator(InsightDecorator):
    def __init__(self, wrapped, upvote_weight=1.0, comment_weight=2.0, sentiment_weight=1.5, sentiment_col_1="title_with_desc_score", sentiment_col_2="comments_score"):
        """
        Initializes the ImportanceScorerDecorator.

        This decorator assigns an importance score to each Reddit post based on engagement metrics 
        (upvotes, downvotes, comments) and sentiment analysis.

        Args:
            wrapped: The base Insight Generator being decorated.
            upvote_weight (float): Weight for the net upvotes (ups - downs).
            comment_weight (float): Weight for the number of comments.
            sentiment_weight (float): Weight for sentiment polarity scores.
        """
        super().__init__(wrapped)
        self.upvote_weight = upvote_weight
        self.comment_weight = comment_weight
        self.sentiment_weight = sentiment_weight
        self.sentiment_col_1 = sentiment_col_1
        self.sentiment_col_2 = sentiment_col_2

    def extract_insights(self, df):
        """
        Enhances insights by calculating and assigning importance scores to each post.

        Args:
            df (pd.DataFrame): DataFrame containing Reddit post data.

        Returns:
            dict: Enhanced insights with importance scores added.

        Raises:
            KeyError: If required columns are missing from the DataFrame.
        """
        insights = self._wrapped.extract_insights(df)

        # Ensure required columns are present in the DataFrame
        required_columns = {"ups", "downs", "num_comments", self.sentiment_col_1, self.sentiment_col_2}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")

        # Fill NaN values with default values
        df["ups"] = df["ups"].fillna(0)
        df["downs"] = df["downs"].fillna(0)
        df["num_comments"] = df["num_comments"].fillna(0)
        df[self.sentiment_col_1] = df[self.sentiment_col_1].fillna(0.0)
        df[self.sentiment_col_2] = df[self.sentiment_col_2].fillna(0.0)

        # Calculate importance scores
        df["importance_score"] = self.calculate_importance_score(df)

        # Log importance scores to a text file
        with open("importance_scores_log.txt", "w", encoding="utf-8") as log_file:
            for idx, score in df["importance_score"].items():
                log_file.write(f"Index: {idx}, Importance Score: {score:.2f}\n")

        # Add importance scores to insights
        insights["importance_scores"] = df[["importance_score"]].to_dict(orient="records")
        return insights

    def calculate_importance_score(self, df):
        """
        Computes the importance score for each post using engagement metrics and sentiment analysis.

        Args:
            df (pd.DataFrame): DataFrame containing Reddit post data.

        Returns:
            pd.Series: A Series containing the calculated importance scores for each post.
        """
        # Calculate the sentiment factor as the sum of absolute sentiment scores
        sentiment_factor = (
            df[self.sentiment_col_1].abs() + df[self.sentiment_col_2].abs()
        )

        # Compute the importance score using the weighted sum of metrics
        importance_score = (
            self.upvote_weight * (df["ups"] - df["downs"]) +
            self.comment_weight * df["num_comments"] +
            self.sentiment_weight * sentiment_factor
        )

        return importance_score.fillna(0)  # Ensure no NaN values in the final output
