import pandas as pd
from insight_generator.base_decorator import InsightDecorator
from insight_generator.insight_interface import InsightGenerator

class SentimentDiscrepancyDecorator(InsightDecorator):
    def __init__(self, wrapped: InsightGenerator, 
                 score_col_1='sentiment', 
                 score_col_2='comments_sentiment', 
                 log_file="sentiment_discrepancies.txt",
                 thresholds=(0.3, 0.5, 0.7)):
        """
        A decorator class to detect and log sentiment discrepancies between two sentiment score columns.
        Discrepancies are categorized into levels (LOW, MED, HIGH) based on defined thresholds.

        :param wrapped: The wrapped InsightGenerator instance.
        :param score_col_1: Name of the first sentiment score column.
        :param score_col_2: Name of the second sentiment score column.
        :param log_file: Path to the file where discrepancies will be logged.
        :param thresholds: Tuple defining thresholds for LOW, MED, and HIGH discrepancy levels.
        """
        super().__init__(wrapped)
        self.score_col_1 = score_col_1
        self.score_col_2 = score_col_2
        self.log_file = log_file
        self.thresholds = thresholds
    
    def extract_insights(self, df: pd.DataFrame):
        """
        Extracts insights and detects sentiment discrepancies in the provided DataFrame.
        Returns a new DataFrame with one row per category, listing post indices for LOW, MED, and HIGH discrepancies.
        """
        insights = self._wrapped.extract_insights(df)

        if 'category' not in df.columns:
            raise ValueError("The DataFrame must contain a 'category' column.")

        with open(self.log_file, "w") as log_file:
            discrepancies = {}

            for i, row in df.iterrows():
                discrepancy_level, diff = self.detect_discrepancy(row)
                if discrepancy_level:
                    post_id = i
                    discrepancies[post_id] = {
                        "discrepancy_level": discrepancy_level,
                        "difference": diff,
                        "category": row['category']
                    }
                    self.log_discrepancy(log_file, post_id, discrepancy_level, diff)

        # Organize discrepancies by category and level
        category_summary = {}

        for post_id, info in discrepancies.items():
            category = info["category"]
            level = info["discrepancy_level"]

            if category not in category_summary:
                category_summary[category] = {
                    "low_indices": [],
                    "med_indices": [],
                    "high_indices": []
                }

            if level == "LOW":
                category_summary[category]["low_indices"].append(post_id)
            elif level == "MED":
                category_summary[category]["med_indices"].append(post_id)
            elif level == "HIGH":
                category_summary[category]["high_indices"].append(post_id)

        # Convert to DataFrame
        result_df = pd.DataFrame([
            {
                "category": category,
                "low_indices": summary["low_indices"],
                "med_indices": summary["med_indices"],
                "high_indices": summary["high_indices"]
            }
            for category, summary in category_summary.items()
        ])

        return result_df

    def detect_discrepancy(self, row):
        """
        Detects sentiment discrepancies for a single row in the DataFrame.

        :param row: A single row from the DataFrame.
        :return: A tuple containing the discrepancy level (LOW, MED, HIGH) and the absolute difference.
        """
        score_1 = row.get(self.score_col_1, 0)
        score_2 = row.get(self.score_col_2, 0)
        diff = abs(score_1 - score_2)

        # Classify discrepancy level based on thresholds
        if diff >= self.thresholds[2]:
            return "HIGH", diff
        elif diff >= self.thresholds[1]:
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
