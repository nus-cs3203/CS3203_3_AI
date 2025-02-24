import pandas as pd
import os
from insight_generator.base_decorator import InsightDecorator
from datetime import datetime
import pytz
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.datematch_decorator import DateMatcherDecorator
from insight_generator.engagement_decorator import EngagementDecorator
from insight_generator.importance_scoring_decorator import ImportanceScoringDecorator
from insight_generator.insight_interface import InsightGenerator
from insight_generator.prompt_decorator import PromptGeneratorDecorator
from insight_generator.sentiment_forecast_decorator import TopicSentimentForecastDecorator

# Load the historical sentiment data (for forecasting)
historical_data = pd.read_csv('files/sentiment_scored_2023_data.csv')  # Make sure this file is available in your environment

# Sample Reddit posts data
posts_data = pd.read_csv('files/sentiment_scored_2023_data.csv').head(10)  # Make sure this file contains the necessary columns like 'Domain Category', 'sentiment_score', etc.

# Initialize the Insight Generator and apply decorators
insight_generator = BaseInsightGenerator()

# Decorate the generator with the different decorators
insight_generator = EngagementDecorator(insight_generator)
insight_generator = ImportanceScoringDecorator(insight_generator)
insight_generator = PromptGeneratorDecorator(insight_generator)
insight_generator = TopicSentimentForecastDecorator(insight_generator, historical_data)

# Function to generate a report
def generate_report(posts_df):
    report = []

    for index, post in posts_df.iterrows():
        insights = insight_generator.extract_insights(post)
        
        # Prepare a summary for the report
        post_report = {
            'Post Title': post['title'],
            'Domain Category': post['Domain Category'],
            'Sentiment': post['sentiment_title_selftext_label'],
            'Sentiment Forecast': insights.get('sentiment_forecast', 'N/A'),
            'Engagement Score': insights['engagement']['engagement_score'],
            'Adjusted Engagement Score': insights['engagement']['adjusted_engagement_score'],
            'Importance Score': insights.get('importance', 'N/A'),
        }
        
        report.append(post_report)
    
    # Convert report to DataFrame for easier inspection
    report_df = pd.DataFrame(report)
    return report_df

# Generate the report based on the posts data
report_df = generate_report(posts_data)

# Save the report to a CSV file
report_df.to_csv('generated_report.csv', index=False)

# Optionally, print the report
print(report_df)
