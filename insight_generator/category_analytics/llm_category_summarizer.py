import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator

class CategorySummarizerDecorator(InsightDecorator):
    def __init__(self, wrapped, text_col=None, category_col="domain_category"):
        super().__init__(wrapped)
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Please set it in your .env file.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

        self.text_col = text_col
        self.category_col = category_col

    def extract_insights(self, df):
        if self.text_col is None:
            if {"title", "description"}.issubset(df.columns):
                df["title_with_desc"] = df["title"].astype(str) + " " + df["description"].astype(str)
                self.text_col = "title_with_desc"
            else:
                raise KeyError("Missing required text columns: title and description")

        summary_data = []
        with open("category_summaries_log.txt", "w", encoding="utf-8") as log_file:
            for category, group in df.groupby(self.category_col):
                combined_text = " ".join(group[self.text_col].dropna().astype(str))
                combined_sentiment = group["sentiment"].mean()
                if combined_text.strip():
                    summary_result = self.generate_summary(combined_text)
                    summary_data.append({
                        self.category_col: category,
                        "summary": summary_result.get("summary", "No summary available"),
                        "concerns": summary_result.get("concerns", []),
                        "suggestions": summary_result.get("suggestions", []),
                        "sentiment": combined_sentiment
                    })
                    log_file.write(f"Category: {category}\nSummary: {summary_result}\n\n")

        return pd.DataFrame(summary_data)

    def generate_summary(self, text):
        user_prompt = f"""
        You are analyzing Reddit posts related to a specific topic. Your task is to extract meaningful insights.

        **Rules to follow:**
        - DO NOT ask for additional input. If the text is unclear or insufficient, return: "Insufficient data to summarize."
        - Focus on **summarizing trends, concerns, and suggestions**.
        - Output must be structured as follows:

        Summary:
        [Brief overview of the discussions]

        Concerns:
        - [Summarized concern]
        - [Summarized concern]

        Suggestions:
        - [Summarized suggestion]
        - [Summarized suggestion]

        **Reddit Posts:**
        {text}

        Generate a concise yet insightful summary based on the above content.
        """
        
        try:
            response = self.model.generate_content(user_prompt)
            result_text = response.text.strip() if response and hasattr(response, 'text') else "No summary generated."
            
            # Ensure correct extraction
            summary = "No summary available"
            concerns = []
            suggestions = []

            
            
            sections = result_text.split("\n\n")  # Split sections by double newlines
            for section in sections:
                if section.startswith("Summary:"):
                    summary = section.replace("Summary:", "").strip()
                elif section.startswith("Concerns:"):
                    concerns = [line.strip("- ") for line in section.split("\n") if line.startswith("-")]
                elif section.startswith("Suggestions:"):
                    suggestions = [line.strip("- ") for line in section.split("\n") if line.startswith("-")]
            
            return {"summary": summary, "concerns": concerns, "suggestions": suggestions}
        except Exception as e:
            return {"summary": "Summary generation failed", "concerns": [], "suggestions": []}