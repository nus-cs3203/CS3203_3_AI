import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator

class PromptGeneratorDecorator(InsightDecorator):
    def __init__(self, wrapped, category_col="Domain Category", log_file="poll_prompts_log.txt"):
        """
        Generates poll prompts based on extracted insights using an LLM.

        Args:
        - wrapped: Base Insight Generator
        - category_col: Column containing categories.
        - log_file: File to log generated poll prompts.
        """
        super().__init__(wrapped)
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Please set it in your .env file.")

        genai.configure(api_key=self.api_key)  # Configure API key
        self.model = genai.GenerativeModel("gemini-2.0-flash")  # Use GenerativeModel
        self.category_col = category_col
        self.log_file = log_file

    def extract_insights(self, df):
        """Processes DataFrame and generates poll prompts per category."""
        insights = self._wrapped.extract_insights(df)

        # Ensure required columns exist
        required_cols = {"title", "selftext", self.category_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Group posts by category and generate poll prompts
        polls_by_category = {}
        log_entries = []
        status_entries = []

        for category, group in df.groupby(self.category_col):
            group["title_selftext"] = group["title"].fillna("") + " " + group["selftext"].fillna("")
            combined_text = " ".join(group["title_selftext"].dropna())

            if combined_text.strip():
                poll_prompt = self.generate_poll_prompt(category, combined_text)
                polls_by_category[category] = poll_prompt
                log_entries.append(f"Category: {category}\nPoll Prompt: {poll_prompt}\n\n")
                status_entries.append(f"{category}: Poll Generated")
            else:
                status_entries.append(f"{category}: No Poll Generated")

        # Write to log file
        if log_entries:
            with open(self.log_file, "a", encoding="utf-8") as log_file:
                log_file.writelines(log_entries)

        # Print summary of poll generation status
        print("\nPoll Generation Status:")
        for status in status_entries:
            print(status)

        insights["polls_by_category"] = polls_by_category
        return insights
    
    def generate_poll_prompt(self, category, text):
        """Uses Gemini API to generate a poll prompt for a given category."""
        user_prompt = f"""
        You are a Reddit poll generator for discussions in the category: **{category}**.

        Based **only** on the following Reddit discussions, generate **one** poll question:

        "{text}"

        **Output Format:**
        ```
        Question: [A concise, relevant question based strictly on the provided discussions]
        Question Type: [MCQ / Single answer / Scale / Open-ended]
        Answers: [Options, if applicable]
        Reason: [Why this poll is useful + trends from given data that makes this post relevant]
        ```

        **Rules:**
        - Stick strictly to the category and the given text.
        - Keep the prompt relevant to the Singapore context
        - Derive prompts from the insights gathered from given text
        - Each statement should stand on its own (i.e. do not refer specific posts or make it too narrow)
        - Avoid unnecessary explanations.
        """

        try:
            response = self.model.generate_content(user_prompt)
            return response.text.strip() if response else "No poll generated."
        except Exception as e:
            return f"Poll generation failed: {str(e)}"
