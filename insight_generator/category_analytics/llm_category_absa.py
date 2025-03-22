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
                absa_result = self.perform_llm_absa(group)  # Pass the DataFrame group
                absa_data.append({
                    self.category_col: category,
                    "absa_result": absa_result["aspects"],
                    "keywords": absa_result["keywords"]
                })
            except Exception as e:
                logging.error(f"Error processing category '{category}': {e}")

        res = pd.DataFrame(absa_data)
        res.dropna(subset=["absa_result", "keywords"], inplace=True)
        res["absa_result"] = res["absa_result"].apply(lambda x: x[:10] if len(x) > 10 else x)
        res["keywords"] = res["keywords"].apply(lambda x: x[:5] if len(x) > 5 else x)
        res.drop_duplicates(subset=[self.category_col], inplace=True)
        logging.info("Finished ABSA extraction.")
        return res

    def perform_llm_absa(self, group):
        """Performs ABSA using the LLM, now accepting a DataFrame."""
        logging.info("Performing ABSA using LLM")
        dataframe_string = group.to_string(index=False)
        user_prompt = f"""
        You are an expert in Aspect-Based Sentiment Analysis (ABSA).
        You will be provided with a table of Reddit discussions related to a specific category.
        Your task is to analyze the discussions and extract key aspects and their sentiments.

        **Input Table:**
        ```
        {dataframe_string}
        ```

        **Instructions:**
        - First, extract exactly 5 keywords that are most relevant to the discussions in the table.
        - Then, extract exactly 10 key aspects and their sentiments from the discussions in the table.
        - Aspects should be high-level concepts, not specific entities (e.g., "food quality" instead of "pizza").
        - Do not have duplicate aspects.
        - Aspects should have at least 2 words.
        - There should be exactly 10 aspects.
        - There should be exactly 5 keywords.
        - Keywords should be single words and important to the text.
        - Do not include any additional information.
        - Focus on extracting key aspects and their sentiments.
        - Skip erroneous or missing categories (or where ABSA failed).
        - This is from Singaporean Reddit threads. Please consider the context.
        - Consider all columns in the table when generating the aspects and keywords.

        **Output Format (strictly follow this format):**
        Keywords 1, Keywords 2, Keywords 3, Keywords 4, Keywords 5
        <Aspect>, positive/neutral/negative
        <Aspect>, positive/neutral/negative
        <Aspect>, positive/neutral/negative
        <Aspect>, positive/neutral/negative
        <Aspect>, positive/neutral/negative
        <Aspect>, positive/neutral/negative
        <Aspect>, positive/neutral/negative
        <Aspect>, positive/neutral/negative
        <Aspect>, positive/neutral/negative
        <Aspect>, positive/neutral/negative

        Sample Response:
        Toyoto, Tesla, Electric, Car, Battery
        Car Price, positive
        Car Performance, positive
        Vehicle Design, neutral
        Battery Life, negative
        Modern Features, positive
        License Accessibility, neutral
        Car Maintenance, negative
        Safety Features, positive
        Brand Reputation, positive
        Easy Charging, neutral
        """
        try:
            response = self.model.generate_content(user_prompt)
            result_text = response.text.strip() if response and hasattr(response, "text") else ""
            lines = result_text.splitlines()
            all_results = []
            keywords = set()
            if lines:
                keyword_line = lines[0]
                keyword_list = keyword_line.replace("List of 5 keywords:", "").strip().split(",")
                for k in keyword_list:
                    keywords.add(k.strip())

                for line in lines[1:]:
                    if "," in line:
                        aspect_sentiment = line.strip()
                        all_results.append(aspect_sentiment)

            return {"aspects": all_results, "keywords": list(keywords)}
        except Exception as e:
            logging.error(f"ABSA failed: {str(e)}")
            return {"aspects": [f"ABSA failed: {str(e)}"], "keywords": []}

    def chunk_text(self, text, max_tokens):
        words = text.split()
        chunk_size = max_tokens // 3
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
