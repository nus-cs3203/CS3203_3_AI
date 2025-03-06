from insight_generator.insight_interface import InsightGenerator

class BaseInsightGenerator(InsightGenerator):
    def extract_insights(self, post):
        return {
            #"name": post["Intent Category"],
            #"summary": post.get("summary", ""),
            "keywords": post.get("keywords", []),
            #"concerns": post.get("concerns", []),
            #"suggestions": post.get("suggestions", []),
            #"sentiment": float(post["sentiment_title_selftext_label"]),
            #"forecasted_sentiment": post.get("forecasted_sentiment", 0.0),
            "dates_with_shift": post.get("dates_with_shift", []),
            #"absa_result": post.get("absa_result", [])
        }
