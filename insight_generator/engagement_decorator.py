from insight_generator.base_decorator import InsightDecorator

class EngagementDecorator(InsightDecorator):
    def extract_insights(self, post):
        insights = super().extract_insights(post)
        insights["engagement"] = {
            "upvotes": post["ups"],
            "downvotes": post["downs"],
            "score": post["score"],
            "upvote_ratio": post["upvote_ratio"],
            "num_comments": post["num_comments"]
        }
        return insights
