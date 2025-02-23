import random
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.engagement_decorator import EngagementDecorator
from insight_generator.importance_scoring_decorator import ImportanceScoringDecorator
from insight_generator.prompt_decorator import PromptGeneratorDecorator
from insight_generator.aggregator_decorator import AggregatorDecorator
from insight_generator.category_summary_decorator import CategoryWiseSummaryDecorator
import pandas as pd

from insight_generator.sentiment_forecast_decorator import TopicSentimentForecastDecorator

# Sample Post (from CSV)
sample_post = {
    "title": "MRT breakdown again, what a mess!",
    "selftext": "Another breakdown during peak hours. Unacceptable!",
    "comments": [
        "This is becoming a regular issue.",
        "I am always late because of these breakdowns.",
        "They need to fix this problem once and for all!"
    ],
    "sentiment_title_selftext_label": "negative",
    "sentiment_title_selftext_polarity": -0.7,
    "sentiment_comments_label": "negative",
    "sentiment_comments_polarity": [-0.5, -0.6, -0.8, 0.1, 0.2],
    "Intent Category": "complaint",
    "Domain Category": "transport",
    "ups": 120,
    "downs": 5,
    "score": 115,
    "upvote_ratio": 0.96,
    "num_comments": 30,
}

# Simulate historical sentiment data for forecasting (this would normally come from your dataset)
historical_data = pd.DataFrame({
    "date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
    "topic": ["transport"] * 100,
    "sentiment_score": [random.uniform(-1, 1) for _ in range(100)]
})

# Base Insight Generator
base_insight = BaseInsightGenerator()

# Apply all decorators
decorated_insight = EngagementDecorator(base_insight)  # Add engagement info
decorated_insight = ImportanceScoringDecorator(decorated_insight)  # Add importance score
decorated_insight = TopicSentimentForecastDecorator(decorated_insight, historical_data)  # Forecast sentiment
decorated_insight = AggregatorDecorator(decorated_insight)  # Combine title & comment sentiment
decorated_insight = CategoryWiseSummaryDecorator(decorated_insight)  # Generate category-wise summary
decorated_insight = PromptGeneratorDecorator(decorated_insight)  # Add prompt generation logic (if needed)

# Extract insights for the sample post
insights = decorated_insight.extract_insights(sample_post)

# Print out the full insights
print(insights)
