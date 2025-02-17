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
        insights = super().extract_insights(post)
        
        # Extract the necessary text components
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        comments = " ".join(post.get("comments", []))  # Join all comments into one string
        
        # Generate summary for title, selftext, and comments
        full_text = f"Title: {title} Selftext: {selftext} Comments: {comments}"
        summary = self.summarize_text(full_text)
        
        # Include sentiment in the summary
        sentiment = post.get("sentiment_title_selftext_label", "neutral")
        
        insights["category_summary"] = {
            "summary": summary,
            "category": post.get("Intent Category", ""),
            "domain": post.get("Domain Category", ""),
            "sentiment": sentiment
        }

        return insights

    def summarize_text(self, text):
        """
        Summarize the given text using the summarization model.
        :param text: The full text to be summarized
        :return: The summarized text
        """
        summary = self.summarizer(text, max_length=200, min_length=50, do_sample=False)
        return summary[0]["summary_text"]
