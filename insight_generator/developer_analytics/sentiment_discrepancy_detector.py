import pandas as pd
from insight_generator.base_decorator import InsightDecorator
from insight_generator.insight_interface import InsightGenerator

class SentimentDiscrepancyDecorator(InsightDecorator):
    def __init__(self, wrapped: InsightGenerator, 
                 score_col_1='title_with_desc_score', 
                 score_col_2='comments_score', 
                 log_file="sentiment_discrepancies.txt",
                 thresholds=(0.3, 0.5, 0.7)):
        """
        Decorator to detect sentiment discrepancies between two sentiment score columns.
        Logs discrepancies classified as LOW, MED, HIGH based on threshold.
        
        :param wrapped: The wrapped InsightGenerator instance.
        :param score_col_1: First sentiment score column.
        :param score_col_2: Second sentiment score column.
        :param log_file: File to log detected discrepancies.
        :param thresholds: Tuple defining LOW, MED, HIGH discrepancy levels.
        """
        super().__init__(wrapped)
        self.score_col_1 = score_col_1
        self.score_col_2 = score_col_2
        self.log_file = log_file
        self.thresholds = thresholds
    
    def extract_insights(self, df: pd.DataFrame):
        """
        Extracts insights including sentiment discrepancy detection.
        :param df: DataFrame containing sentiment data
        :return: Insights with sentiment discrepancy results
        """
        insights = self._wrapped.extract_insights(df)

        # Open log file once for efficiency
        with open(self.log_file, "w") as log_file:
            discrepancies = {}

            for i, row in df.iterrows():
                discrepancy_level, diff = self.detect_discrepancy(row)
                if discrepancy_level:
                    post_id = i
                    discrepancies[post_id] = {
                        "discrepancy_level": discrepancy_level,
                        "difference": diff
                    }
                    self.log_discrepancy(log_file, post_id, discrepancy_level, diff)

        insights["sentiment_discrepancies"] = discrepancies
        return insights

    def detect_discrepancy(self, row):
        """
        Detects sentiment discrepancies for a given row.
        :param row: A single row from the DataFrame
        :return: Discrepancy level (LOW, MED, HIGH) and difference
        """
        score_1 = row.get(self.score_col_1, 0)
        score_2 = row.get(self.score_col_2, 0)
        diff = abs(score_1 - score_2)

        # Classify discrepancy level
        if diff >= self.thresholds[2]:
            return "HIGH", diff
        elif diff >= self.thresholds[1]:
            return "MED", diff
        elif diff >= self.thresholds[0]:
            return "LOW", diff
        else:
            return None, diff

    def log_discrepancy(self, log_file, post_id, level, diff):
        """
        Logs detected sentiment discrepancies to a file.
        :param log_file: Open file handle for logging
        :param post_id: Post ID where discrepancy was found
        :param level: Discrepancy level (LOW, MED, HIGH)
        :param diff: Absolute difference between sentiment scores
        """
        log_file.write(f"Post Index: {post_id} | Diff: {diff:.2f} | Level: {level}\n")
