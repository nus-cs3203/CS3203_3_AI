import pandas as pd
from transformers import pipeline
from insight_generator.base_decorator import InsightDecorator

class CategoryWiseSummaryDecorator(InsightDecorator):
    def __init__(self, wrapped_insight_generator, summarizer=None):
        """
        Summarizes content from the title, selftext, and comments based on category and sentiment.
        :param wrapped_insight_generator: The base insight generator to wrap
        :param summarizer: The summarizer model (default: BART)
        """
        super().__init__(wrapped_insight_generator)
        self.summarizer = summarizer or pipeline("summarization", model="facebook/bart-large-cnn")

    def extract_insights(self, post):
        insights_list = []

        for _, row in post.iterrows():
            insights = super().extract_insights(row)
            title = row.get("title", "")
            selftext = row.get("selftext", "")
            comments = " ".join(row.get("comments", []))  # Join all comments into one string
            full_text = f"Title: {title} Selftext: {selftext} Comments: {comments}"

            # Directly perform summarization in-line
            summary = self.summarizer(full_text, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
            
            sentiment = row.get("sentiment_title_selftext_label", "neutral")
            
            insights["category_summary"] = {
                "summary": summary,
                "category": row.get("Intent Category", ""),
                "domain": row.get("Domain Category", ""),
                "sentiment": sentiment
            }

            insights_list.append(insights)
        
        return pd.DataFrame(insights_list)
