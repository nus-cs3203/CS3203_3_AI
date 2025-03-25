import os
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator

class ImportanceScorerDecorator(InsightDecorator):
    def __init__(self, wrapped, upvote_weight=1.0, comment_weight=2.0, sentiment_weight=1.5):
        """
        Assigns an importance score to each Reddit post based on engagement parameters.

        Args:
        - wrapped: Base Insight Generator.
        - upvote_weight: Weight for (ups - downs).
        - comment_weight: Weight for number of comments.
        - sentiment_weight: Weight for sentiment polarity.
        """
        super().__init__(wrapped)
        self.upvote_weight = upvote_weight
        self.comment_weight = comment_weight
        self.sentiment_weight = sentiment_weight

    def extract_insights(self, df):
        """Enhances insights by assigning importance scores to each post."""
        insights = self._wrapped.extract_insights(df)

        required_columns = {"ups", "downs", "num_comments", "title_with_desc_score", "comments_score"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")

        # Fill NaN values with defaults
        df["ups"] = df["ups"].fillna(0)
        df["downs"] = df["downs"].fillna(0)
        df["num_comments"] = df["num_comments"].fillna(0)
        df["title_with_desc_score"] = df["title_with_desc_score"].fillna(0.0)
        df["comments_score"] = df["comments_score"].fillna(0.0)

        df["importance_score"] = self.calculate_importance_score(df)

        # Save results in a txt file with index and scores
        with open("importance_scores_log.txt", "w", encoding="utf-8") as log_file:
            for idx, score in df["importance_score"].items():
                log_file.write(f"Index: {idx}, Importance Score: {score:.2f}\n")

        insights["importance_scores"] = df[["importance_score"]].to_dict(orient="records")
        return insights

    def calculate_importance_score(self, df):
        """Computes the importance score using engagement metrics and sentiment analysis."""
        sentiment_factor = (
            df["title_with_desc_score"].abs() + df["comments_score"].abs()
        )

        importance_score = (
            self.upvote_weight * (df["ups"] - df["downs"]) +
            self.comment_weight * df["num_comments"] +
            self.sentiment_weight * sentiment_factor
        )

        return importance_score.fillna(0)  # Ensure no NaN values in final output
