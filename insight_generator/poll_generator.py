import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

class PollGenerator():
    def __init__(self, category_col="category", log_file="poll_prompts_log.txt"):
        """
        Generates poll prompts based on extracted insights using an LLM.

        Args:
        - category_col: Column containing categories.
        - log_file: File to log generated poll prompts.
        """
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

        # Ensure required columns exist
        required_cols = {"title", "description", self.category_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Group posts by category and generate poll prompts
        polls_data = []

        for category, group in df.groupby(self.category_col):
            if len(group) < 10:
                print(f"Skipping category '{category}' because it has fewer than 10 posts.")
                continue  # Skip categories with less than 10 posts

            # Pass the group DataFrame to generate_poll_prompt
            poll_prompt = self.generate_poll_prompt(category, group)
            polls_data.append({
                "category": category,
                "question": poll_prompt.get("question", ""),
                "question_type": poll_prompt.get("question_type", ""),
                "options": poll_prompt.get("options", []),
                "reasoning": poll_prompt.get("reasoning", "")
            })

        insights = pd.DataFrame(polls_data)
        insights.dropna(inplace=True)
        return insights

    def generate_poll_prompt(self, category, group):
        """Uses Gemini API to generate a poll prompt for a given category, now with DataFrame input."""
        # Convert the DataFrame to a string representation for the LLM
        dataframe_string = group.to_string(index=False)

        user_prompt = f"""
        You are a Reddit poll generator for discussions in the category: **{category}**.

        Based **only** on the following Reddit discussions (provided as a table), generate **one** poll question for category **{category}**:

        ```
        {dataframe_string}
        ```

        **Output Format (each in new line, without any extra formatting or markdown like ```):**
        [A concise, relevant question based strictly on the provided discussions]
        [MCQ / Open-ended]
        [Options, if applicable, separated by semicolons]
        [Why this poll is useful + trends from given data that makes this post relevant]

        **Rules:**
        - Stick strictly to the category and the given data.
        - Generate a single poll question.
        - Keep the prompt relevant to the Singapore context
        - Derive prompts from the insights gathered from given data
        - Reasoning should be based on the trends observed in the data
        - Reasoning should be atleast 10 words long
        - MCQ options should be descriptive and relevant to the category
        - Each statement should stand on its own (i.e. do not refer specific posts or make it too narrow)
        - Avoid unnecessary explanations.
        - Consider all columns in the table when generating the poll.

        Sample Output:
        What is the most popular food in Singapore?
        MCQ
        Chicken Rice; Laksa; Hainanese Curry Rice; Nasi Lemak
        This poll is useful to understand the most popular food choices in Singapore. The given discussions show a trend towards local cuisine, making this post relevant.
        """

        try:
            response = self.model.generate_content(user_prompt)
            if response:
                # Parse the response text to extract the required fields
                lines = [line.strip() for line in response.text.strip().split("\n") if line.strip() and not line.startswith("```")]

                return {
                    "question": lines[0] if len(lines) > 0 else "",
                    "question_type": lines[1] if len(lines) > 1 else "",
                    "options": lines[2].split(";") if len(lines) > 2 else [],
                    "reasoning": lines[3] if len(lines) > 3 else ""
                }
            else:
                return {
                    "question": "No poll generated.",
                    "question_type": "",
                    "options": [],
                    "reasoning": ""
                }
        except Exception as e:
            return {
                "question": f"Poll generation failed: {str(e)}",
                "question_type": "",
                "options": [],
                "reasoning": ""
            }
