import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator

class CategorySummarizerDecorator(InsightDecorator):
    def __init__(self, wrapped, text_col="title_selftext", category_col="Domain Category"):
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

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-pro")  # Initialize once

        self.text_col = text_col
        self.category_col = category_col

    def extract_insights(self, df):
        """Processes the DataFrame and generates category-level summaries using LLM."""
        insights = self._wrapped.extract_insights(df)

        # Group text by category
        category_summaries = {}
        with open("category_summaries_log.txt", "w", encoding="utf-8") as log_file:
            for category, group in df.groupby(self.category_col):
                combined_text = " ".join(group[self.text_col].dropna())  # Concatenate all text

                if combined_text.strip():  # Avoid empty summaries
                    summary = self.generate_summary(combined_text)  # LLM Call
                    
                    category_summaries[category] = summary
                    log_file.write(f"Category: {category}\nSummary: {summary}\n\n")

        insights["category_summaries"] = category_summaries
        return insights

    def generate_summary(self, text):
        """Uses Gemini API to summarize key concerns."""
        user_prompt = f"""
        Summarize key concerns and discussion points from the following Reddit posts:

        {text}

        The summary should highlight:
        - **Main topics discussed**.
        - **Common concerns raised by users**.
        - **Any trends or patterns in sentiment**.
        - **Notable suggestions or feedback**.

        Output in a structured format:
        ```
        Key Concerns:
        - [Concern 1]
        - [Concern 2]

        Suggestions:
        - [Suggestion 1]
        - [Suggestion 2]
        ```
        """

        try:
            response = self.model.generate_content(user_prompt)
            return response.text.strip() if response else "No summary generated."
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
