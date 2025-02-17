import os
import google.generativeai as genai

from insight_generator.base_decorator import InsightDecorator

class PromptGeneratorDecorator(InsightDecorator):
    def __init__(self, wrapped):
        super().__init__(wrapped)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Load API key

    def extract_insights(self, post):
        insights = super().extract_insights(post)

        # Ensure required keys exist
        insights.setdefault("Intent Category", "unknown")
        insights.setdefault("Domain Category", "general")
        insights.setdefault("sentiment_title_selftext_label", "neutral")
        insights.setdefault("score", 0)
        insights.setdefault("ups", 0)
        insights.setdefault("downs", 0)

        # Generate Yes/No or Agree/Disagree poll
        insights["generated_poll"] = self.generate_poll_prompt(insights)
        return insights

    def generate_poll_prompt(self, insights):
        user_prompt = f"""
        Generate a **Yes/No or Agree/Disagree** polling question based on the following Reddit insight:

        - **Intent**: {insights.get('Intent Category', 'unknown')}
        - **Domain**: {insights.get('Domain Category', 'general')}
        - **Sentiment**: {insights.get('sentiment_title_selftext_label', 'neutral')}
        - **Post Score**: {insights.get('score', 0)}
        - **Engagement (Ups: {insights.get('ups', 0)}, Downs: {insights.get('downs', 0)})**

        The poll should:
        1. Be **short and clear** (one sentence).
        2. Use **Yes/No** or **Agree/Disagree**.
        3. Be relevant for authorities to take action on.

        Example output:
        ```
        Poll: [Generated question]
        👍 Yes  
        👎 No  
        ```
        """

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(user_prompt)
        
        return response.text.strip() if response else "No poll generated."
