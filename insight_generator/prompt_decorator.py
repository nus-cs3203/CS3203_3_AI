import os
import google.generativeai as genai
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator

class PromptGeneratorDecorator(InsightDecorator):
    def __init__(self, wrapped, time_window_days=7, category_col="Domain Category"):
        """
        Generates poll prompts based on extracted insights using an LLM.

        Args:
        - wrapped: Base Insight Generator
        - time_window_days: Filters posts within the last X days.
        - category_col: Column containing categories.
        """
        super().__init__(wrapped)
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Please set it in your .env file.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-pro")  # Initialize once
        self.time_window_days = time_window_days
        self.category_col = category_col

    def extract_insights(self, df):
        """Processes DataFrame and generates poll prompts per category."""
        insights = self._wrapped.extract_insights(df)

        # Filter posts within the time window
        time_cutoff = datetime.utcnow() - timedelta(days=self.time_window_days)
        df = df[pd.to_datetime(df["utc_created_at"], unit="s") >= time_cutoff]

        # Group posts by category and generate poll prompts
        polls_by_category = {}
        with open("poll_prompts_log.txt", "w", encoding="utf-8") as log_file:
            for category, group in df.groupby(self.category_col):
                combined_text = " ".join(group["title_selftext"].dropna())  # Concatenate all text

                if combined_text.strip():  # Avoid empty polls
                    poll_prompt = self.generate_poll_prompt(category, combined_text)  # LLM Call
                    
                    polls_by_category[category] = poll_prompt
                    log_file.write(f"Category: {category}\nPoll Prompt: {poll_prompt}\n\n")

        insights["polls_by_category"] = polls_by_category
        return insights

    def generate_poll_prompt(self, category, text):
        """Uses Gemini API to generate a poll prompt for a given category."""
        user_prompt = f"""
        Based on the following Reddit discussions in the category **{category}**, generate a poll question:

        {text}

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
