from insight_generator.insight_interface import InsightGenerator

class BaseInsightGenerator(InsightGenerator):
    def extract_insights(self, post):
        return {
            "name": post["category"],
            "summary": post.get("summary", ""),
            "keywords": post.get("keywords", []),
            "concerns": post.get("concerns", []),
            "suggestions": post.get("suggestions", []),
            "sentiment": float(post["sentiment"]),
            "forecasted_sentiment": post.get("forecasted_sentiment", 0.0),
            "absa_result": post.get("absa_result", [])
        }
