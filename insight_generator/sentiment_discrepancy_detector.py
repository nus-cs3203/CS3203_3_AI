import pandas as pd
from insight_generator.base_decorator import InsightDecorator
from insight_generator.insight_interface import InsightGenerator

class SentimentDiscrepancyDecorator(InsightDecorator):
    def __init__(self, wrapped: InsightGenerator, 
                 score_col_1='sentiment_title_selftext_polarity', 
                 score_col_2='sentiment_comments_polarity', 
                 thresholds=(0.3, 0.5, 0.7)):
        """
        Decorator to detect sentiment discrepancies between two sentiment score columns.
        Logs discrepancies classified as LOW, MED, HIGH based on threshold.
        
        :param wrapped: The wrapped InsightGenerator instance.
        :param score_col_1: First sentiment score column.
        :param score_col_2: Second sentiment score column.
        :param thresholds: Tuple defining LOW, MED, HIGH discrepancy levels.
        """
        super().__init__(wrapped)
        self.score_col_1 = score_col_1
        self.score_col_2 = score_col_2
        self.thresholds = thresholds
    
    def extract_insights(self, post):
        insights = self._wrapped.extract_insights(post)
        
        # Extract sentiment scores
        score_1 = post.get(self.score_col_1, 0)
        score_2 = post.get(self.score_col_2, 0)
        
        # Calculate absolute difference
        diff = abs(score_1 - score_2)
        
        # Classify discrepancy level
        if diff >= self.thresholds[2]:
            discrepancy_level = "HIGH"
        elif diff >= self.thresholds[1]:
            discrepancy_level = "MED"
        elif diff >= self.thresholds[0]:
            discrepancy_level = "LOW"
        else:
            discrepancy_level = None
        
        # Log and add to insights if discrepancy exists
        if discrepancy_level:
            insights["sentiment_discrepancy"] = discrepancy_level
            self._log_discrepancy(post, discrepancy_level, diff)
        
        return insights
    
    def _log_discrepancy(self, post, level, diff):
        """Logs sentiment discrepancies into a text file named after the decorator."""
        log_entry = (f"Post ID: {post.get('id', 'N/A')} | "
                     f"Diff: {diff:.2f} | "
                     f"Level: {level}\n")
        with open("sentiment_discrepancy_log.txt", "a") as log_file:
            log_file.write(log_entry)
