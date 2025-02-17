# Example Reddit post (simulate dataset row)
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.engagement_decorator import EngagementDecorator
from insight_generator.importance_scoring_decorator import ImportanceScoringDecorator
from insight_generator.prompt_decorator import PromptGeneratorDecorator


sample_post = {
    "title": "MRT breakdown again, what a mess!",
    "selftext": "Another breakdown during peak hours. Unacceptable!",
    "ups": 120,
    "downs": 5,
    "score": 115,
    "upvote_ratio": 0.96,
    "num_comments": 30,
    "sentiment_title_selftext_label": "negative",
    "Intent Category": "complaint",
    "Domain Category": "transport"
}

if __name__ == "__main__":
    base_insight = BaseInsightGenerator()
    
    # Apply decorators
    decorated_insight = EngagementDecorator(base_insight)
    decorated_insight = PromptGeneratorDecorator(decorated_insight)
    decorated_insight = ImportanceScoringDecorator(decorated_insight)
    
    insights = decorated_insight.extract_insights(sample_post)
    print(insights)
