import os
import google.generativeai as genai
import pandas as pd
from datetime import datetime, timedelta
from insight_generator.base_decorator import InsightDecorator
from dotenv import load_dotenv

class PromptGeneratorDecorator(InsightDecorator):
    def __init__(self, wrapped, time_window_days=7, category_filter=None):
        """
        Initializes the Prompt Generator with optional time-based and category-based filtering.
        
        Args:
        - wrapped: Base Insight Generator
        - time_window_days: Integer, filters posts within the last X days (default=7)
        - category_filter: String, filters by category (default=None, meaning all categories)
        """
        super().__init__(wrapped)
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Please set it in your .env file.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-pro")  # Initialize once
        self.time_window_days = time_window_days
        self.category_filter = category_filter.lower() if category_filter else None
        self.polls_by_category = {}

    def extract_insights(self, post):
        """Extracts insights and generates a poll question for eligible posts."""
        insights = super().extract_insights(post)

        # Ensure required keys exist
        insights.setdefault("category", "general")
        insights.setdefault("utc_created_at", 0)  # Unix timestamp
        insights.setdefault("score", 0)
        insights.setdefault("ups", 0)
        insights.setdefault("downs", 0)

        # Check if the post falls within the time window
        post_time = datetime.utcfromtimestamp(insights["utc_created_at"])
        time_cutoff = datetime.utcnow() - timedelta(days=self.time_window_days)
        if post_time < time_cutoff:
            return insights  # Skip poll generation if post is too old

        # Check if category filtering is applied
        category = insights["category"].lower()
        if self.category_filter and category != self.category_filter:
            return insights  # Skip if the category doesn't match

        # Generate poll question for eligible posts
        poll_question = self.generate_poll_prompt(insights)
        if poll_question:
            if category not in self.polls_by_category:
                self.polls_by_category[category] = []
            self.polls_by_category[category].append(poll_question)

        return insights

    def generate_poll_prompt(self, insights):
        """Generates a poll question using the Gemini API."""
        user_prompt = f"""
        Generate a poll question based on the following Reddit insight:

        - **Category**: {insights["category"]}
        - **Sentiment**: {insights.get("sentiment_title_selftext_label", "neutral")}
        - **Post Score**: {insights["score"]}
        - **Engagement (Ups: {insights["ups"]}, Downs: {insights["downs"]})**
        - **Time of Post**: {datetime.utcfromtimestamp(insights["utc_created_at"]).strftime('%Y-%m-%d %H:%M:%S')} UTC

        The poll should include:
        1. **Question**: A **clear and concise** poll question.
        2. **Type**: Specify if it's:
            - **MCQ** (Multiple Choice)
            - **Single answer** (Yes/No or Agree/Disagree)
            - **Scale** (1-5 rating)
            - **Open-ended** (User provides a response)
        3. **Answers** (if applicable).
        4. **Reasoning**: Why this poll is useful.

        Example output:
        ```
        Question: [Generated question]
        Question Type: [Type]
        Answers: [Options, if applicable]
        Reason: [Explanation]
        ```
        """

        try:
            response = self.model.generate_content(user_prompt)
            return response.text.strip() if response else "No poll generated."
        except Exception as e:
            return f"Poll generation failed: {str(e)}"

    def process_batch(self, posts_df):
        """Processes a batch of posts, filtering by time and category before generating polls."""
        posts_df.apply(self.extract_insights, axis=1)
        self.generate_report()

    def generate_report(self):
        """Creates a report summarizing generated polls per category."""
        with open("report.txt", "w", encoding="utf-8") as report_file:
            report_file.write("Poll Generation Report\n")
            report_file.write("=======================\n\n")
            for category, polls in self.polls_by_category.items():
                report_file.write(f"Category: {category}\n")
                report_file.write(f"Total Polls: {len(polls)}\n\n")
                for i, poll in enumerate(polls, 1):
                    report_file.write(f"Poll {i}:\n{poll}\n\n")
                report_file.write("----------------------------------\n\n")
