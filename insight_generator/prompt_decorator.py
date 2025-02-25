import os
import google.generativeai as genai

from insight_generator.base_decorator import InsightDecorator
from dotenv import load_dotenv

class PromptGeneratorDecorator(InsightDecorator):
    def __init__(self, wrapped):
        super().__init__(wrapped)
        # Load environment variables from .env file
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        genai.configure(api_key=GOOGLE_API_KEY)  # Load API key

    def extract_insights(self, post):
        insights = super().extract_insights(post)

        # Ensure required keys exist
        insights.setdefault("Intent Category", "unknown")
        insights.setdefault("Domain Category", "general")
        insights.setdefault("sentiment_title_selftext_label", "neutral")
        insights.setdefault("score", 0)
        insights.setdefault("ups", 0)
        insights.setdefault("downs", 0)

        # Generate poll details
        insights["generated_poll"] = self.generate_poll_prompt(insights)
        return insights

    def generate_poll_prompt(self, insights):
        user_prompt = f"""
        Generate a poll question based on the following Reddit insight:

        - **Intent**: {insights.get('Intent Category', 'unknown')}
        - **Domain**: {insights.get('Domain Category', 'general')}
        - **Sentiment**: {insights.get('sentiment_title_selftext_label', 'neutral')}
        - **Post Score**: {insights.get('score', 0)}
        - **Engagement (Ups: {insights.get('ups', 0)}, Downs: {insights.get('downs', 0)})**

        The poll should include the following:
        1. **Question**: A **clear and concise** poll question that aligns with the post's context.
        2. **Question Type**: Specify if the poll is:
            - **MCQ** (Multiple Choice with 1 correct answer)
            - **Single answer** (Yes/No or Agree/Disagree)
            - **Scale** (e.g., Rate from 1-5, etc.)
            - **Open-ended** (User provides a response)
        3. **Answers** (if applicable):
            - For **MCQ**: Provide 3-5 answer choices.
            - For **Single answer**: Yes/No or Agree/Disagree.
            - For **Scale**: Specify the range of scores (e.g., 1-5).
            - For **Open-ended**: Allow the user to give any response.
        4. **Reasoning**: Why this poll is relevant and important for authorities to consider action.
        
        Example output:
        ```
        Question: [Generated question]
        Question Type: [Type]
        Answers: [Answer choices, if applicable]
        Reason: [Explanation]
        ```
        """

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(user_prompt)
        
        return response.text.strip() if response else "No poll generated."
