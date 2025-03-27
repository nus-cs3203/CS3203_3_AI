from insight_generator.insight_interface import InsightGenerator

class BaseInsightDeveloperGenerator(InsightGenerator):
    def extract_insights(self, post):
        return {
            "name": post.get("category", ""),
            "dates_with_shift": post.get("dates_with_shift", []),
            "absa_result": post.get("absa_result", []),
            "sentiment_clusters": post.get("sentiment_clusters", []),
            "sentiment_discrepancies": post.get("sentiment_discrepancies", []),
            "importance_score": post.get("importance_score", 0.0),
        }
