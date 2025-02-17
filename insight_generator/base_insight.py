from insight_generator.insight_interface import InsightGenerator

class BaseInsightGenerator(InsightGenerator):
    def extract_insights(self, post):
        return {
            "title": post["title"],
            "selftext": post["selftext"],
            "sentiment": post["sentiment_title_selftext_label"],
            "intent": post["Intent Category"],
            "domain": post["Domain Category"],
            "metadata": {
                "num_comments": post["num_comments"],
                "score": post["score"]
            }
        }
