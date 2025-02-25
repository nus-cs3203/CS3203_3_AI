import pandas as pd
from insight_generator.base_decorator import InsightDecorator
from insight_generator.insight_interface import InsightGenerator

class AggregatorDecorator(InsightDecorator):
    def __init__(
        self, wrapped: InsightGenerator, 
        sentiment_cols=None, 
        sentiment_threshold=0.5,
        report_path="report.txt"
    ):
        """
        Aggregates sentiment scores from multiple columns and highlights significant differences.
        :param wrapped: The base insight generator to wrap.
        :param sentiment_cols: Dictionary specifying column names:
            {
                "title": "title_with_desc_score",
                "comments": "sentiment_score_comments"
            }
        :param sentiment_threshold: The threshold for sharp sentiment differences.
        :param report_path: Path to save the generated report.
        """
        super().__init__(wrapped)
        self.sentiment_cols = sentiment_cols if sentiment_cols else {
            "title": "title_with_desc_score",
            "comments": "sentiment_score_comments"
        }
        self.sentiment_threshold = sentiment_threshold
        self.report_path = report_path
        self.analysis_results = []  # Store results for report

    def extract_insights(self, post):
        insights = super().extract_insights(post)  # Get base insights

        # Extract sentiment scores using configurable column names
        title_sentiment_score = post.get(self.sentiment_cols.get("title", ""), 0)
        comment_sentiment_score = post.get(self.sentiment_cols.get("comments", ""), 0)

        # Calculate sentiment difference
        sentiment_diff = abs(title_sentiment_score - comment_sentiment_score)
        sharp_diff_alert = sentiment_diff > self.sentiment_threshold

        # Store sentiment data
        insights["combined_sentiment"] = {
            "title_sentiment": title_sentiment_score,
            "comment_sentiment": comment_sentiment_score,
            "sentiment_diff": sentiment_diff
        }
        insights["sharp_diff_alert"] = sharp_diff_alert

        # Save analysis result for reporting
        self.analysis_results.append({
            "title_sentiment": title_sentiment_score,
            "comment_sentiment": comment_sentiment_score,
            "sentiment_diff": sentiment_diff,
            "sharp_diff_alert": sharp_diff_alert
        })
        
        return insights

    def generate_report(self):
        """Generate a detailed report on sentiment aggregation."""
        total_posts = len(self.analysis_results)
        if total_posts == 0:
            report_content = "No posts were processed."
        else:
            avg_title_sentiment = sum(d["title_sentiment"] for d in self.analysis_results) / total_posts
            avg_comment_sentiment = sum(d["comment_sentiment"] for d in self.analysis_results) / total_posts
            sharp_diffs = sum(1 for d in self.analysis_results if d["sharp_diff_alert"])
            sharp_diff_percentage = (sharp_diffs / total_posts) * 100

            report_content = f"""
            Sentiment Aggregation Report
            ---------------------------------
            Total Posts Analyzed: {total_posts}
            Average Title Sentiment Score: {avg_title_sentiment:.2f}
            Average Comment Sentiment Score: {avg_comment_sentiment:.2f}
            Number of Posts with Sharp Sentiment Difference: {sharp_diffs}
            Percentage of Sharp Sentiment Differences: {sharp_diff_percentage:.2f}%
            """

        # Save report to file
        with open(self.report_path, "w") as report_file:
            report_file.write(report_content)
        print(f"Report saved to {self.report_path}")
