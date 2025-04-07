import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator
import logging

# Configure logging
logging.basicConfig(filename='category_absa.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class CategoryABSAWithLLMInsightDecorator(InsightDecorator):
    def __init__(self, wrapped, text_col=None, category_col="category", max_tokens=3000):
        super().__init__(wrapped)
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Please set it in your .env file.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.text_col = text_col
        self.category_col = category_col
        self.max_tokens = max_tokens

    def extract_insights(self, df):
        logging.info("Starting ABSA extraction.")
        if self.text_col is None:
            if {"title", "description"}.issubset(df.columns):
                df["title_with_desc"] = df["title"].astype(str) + " " + df["description"].astype(str)
                self.text_col = "title_with_desc"
            else:
                raise KeyError("Missing required text columns: title and description")

        absa_data = []
        for category, group in df.groupby(self.category_col):
            logging.info(f"Processing category: {category}")
            try:
                keywords = self.perform_llm_keywords_extraction(group)
                aspects = self.perform_llm_aspect_sentiment_analysis(group)
                absa_data.append({
                    self.category_col: category,
                    "absa_result": aspects,
                    "keywords": keywords
                })
            except Exception as e:
                logging.error(f"Error processing category '{category}': {e}")

        res = pd.DataFrame(absa_data)
        res.dropna(inplace=True)
        res["absa_result"] = res["absa_result"].apply(lambda x: x[:10] if len(x) > 10 else x)
        res["keywords"] = res["keywords"].apply(lambda x: x[:5] if len(x) > 5 else x)
        logging.info("Finished ABSA extraction.")
        return res

    def perform_llm_keywords_extraction(self, group):
        logging.info("Extracting keywords using LLM")
        dataframe_string = group.to_string(index=False)
        user_prompt = f"""
        You are an expert text analyst. Below is a table of Reddit discussions on a specific category.

        **Input Table:**
        ```
        {dataframe_string}
        ```

        **Instructions:**
        - Extract exactly 8 **single-word keywords** most relevant to the discussions.
        - Focus on nouns and important concepts only.
        - Do not repeat words or include phrases.
        - No extra explanation or commentary.

        **Output Format (strictly this line only):**
        keyword1, keyword2, keyword3, keyword4, keyword5, keyword6, keyword7, keyword8
        """

        try:
            response = self.model.generate_content(user_prompt)
            result_text = response.text.strip()
            keyword_line = result_text.splitlines()[0]
            return [kw.strip() for kw in keyword_line.split(",") if kw.strip()]
        except Exception as e:
            logging.error(f"Keyword extraction failed: {str(e)}")
            return []

    def perform_llm_aspect_sentiment_analysis(self, group):
        logging.info("Extracting aspects and sentiments using LLM")
        dataframe_string = group.to_string(index=False)
        user_prompt = f"""
        You are an expert in Aspect-Based Sentiment Analysis (ABSA).
        Below is a table of Reddit discussions.

        **Input Table:**
        ```
        {dataframe_string}
        ```

        **Instructions:**
        - Extract exactly 8 **distinct high-level aspects** (at least 2 words each).
        - For each aspect, assign a sentiment: positive, neutral, or negative.
        - No duplicate aspects.
        - No explanation or extra output.

        **Output Format (each on new line):**
        <Aspect>, positive/neutral/negative

        Example:
        Food Quality, positive
        Customer Service, negative
        """

        try:
            response = self.model.generate_content(user_prompt)
            result_text = response.text.strip()
            lines = result_text.splitlines()
            aspects = [line.strip() for line in lines if "," in line]
            return aspects
        except Exception as e:
            logging.error(f"Aspect sentiment analysis failed: {str(e)}")
            return [f"ABSA failed: {str(e)}"]
