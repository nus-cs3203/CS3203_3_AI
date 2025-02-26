import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator

class CategorySummarizerDecorator(InsightDecorator):
    def __init__(self, wrapped, text_col=None, category_col="Domain Category"):
        """
        Summarizes discussions for each category using an LLM API.

        Args:
        - wrapped: Base Insight Generator
        - text_col: Column containing text to summarize.
        - category_col: Column containing categories.
        """
        super().__init__(wrapped)
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Please set it in your .env file.")

        genai.configure(api_key=self.api_key)  # Configure API key
        self.model = genai.GenerativeModel("gemini-2.0-flash")  # Use GenerativeModel

        self.text_col = text_col
        self.category_col = category_col

    def extract_insights(self, df):
        """Processes the DataFrame and generates category-level summaries using LLM."""
        insights = self._wrapped.extract_insights(df)

        # Ensure text_col is set
        if self.text_col is None:
            if "title" in df.columns and "selftext" in df.columns:
                df["title_selftext"] = df["title"].astype(str) + " " + df["selftext"].astype(str)
                self.text_col = "title_selftext"
            else:
                raise KeyError("Missing required text columns: title and selftext")

        # Group text by category
        category_summaries = {}
        with open("category_summaries_log.txt", "w", encoding="utf-8") as log_file:
            for category, group in df.groupby(self.category_col):
                combined_text = " ".join(group[self.text_col].dropna().astype(str))  # Ensure text is string

                if combined_text.strip():  # Avoid empty summaries
                    summary = self.generate_summary(combined_text)  # LLM Call
                    category_summaries[category] = summary
                    log_file.write(f"Category: {category}\nSummary: {summary}\n\n")

        insights["category_summaries"] = category_summaries
        return insights

    def generate_summary(self, text):
        """Uses Gemini API to summarize key concerns while ensuring structured output."""
        user_prompt = f"""
        You are analyzing Reddit posts related to a specific topic. Your task is to extract meaningful insights.

        **Rules to follow:**
        - DO NOT ask for additional input. If the text is unclear or insufficient, return: "Insufficient data to summarize."
        - Focus on **summarizing trends, concerns, and suggestions**.
        - Output must be structured as follows:

        ```
        Key Concerns:
        - [Summarized concern]
        - [Summarized concern]

        Suggestions:
        - [Summarized suggestion]
        - [Summarized suggestion]
        ```

        **Reddit Posts:**
        {text}

        Generate a concise yet insightful summary based on the above content.
        """

        try:
            response = self.model.generate_content(user_prompt)
            return response.text.strip() if response and hasattr(response, 'text') else "No summary generated."
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
