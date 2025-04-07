import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator
import logging
import requests
from newsapi import NewsApiClient


# Configure logging
logging.basicConfig(filename='category_absa.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class CategoryABSAWithLLMInsightDecorator(InsightDecorator):
    """
    A decorator class for performing Aspect-Based Sentiment Analysis (ABSA) 
    using a Large Language Model (LLM) on categorized data.

    Attributes:
        wrapped (InsightDecorator): The wrapped decorator object.
        text_col (str): The column containing text data for analysis.
        category_col (str): The column containing category labels.
        max_tokens (int): The maximum number of tokens for LLM input.
    """
    def __init__(self, wrapped, text_col=None, category_col="category", max_tokens=3000):
        """
        Initializes the decorator with the required configurations.

        Args:
            wrapped (InsightDecorator): The wrapped decorator object.
            text_col (str, optional): The column containing text data. Defaults to None.
            category_col (str): The column containing category labels. Defaults to "category".
            max_tokens (int): The maximum number of tokens for LLM input. Defaults to 3000.
        """
        super().__init__(wrapped)
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Please set it in your .env file.")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        if not self.news_api_key:
            raise ValueError("NEWS_API_KEY is missing. Please set it in your .env file.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.newsapi = NewsApiClient(api_key=self.news_api_key)
        self.text_col = text_col
        self.category_col = category_col
        self.max_tokens = max_tokens

    def extract_insights(self, df):
        """
        Extracts insights from the given DataFrame by performing ABSA.

        Args:
            df (pd.DataFrame): The input DataFrame containing text and category data.

        Returns:
            pd.DataFrame: A DataFrame containing extracted aspects and keywords for each category.
        """
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
                })
            except Exception as e:
                logging.error(f"Error processing category '{category}': {e}")

        res = pd.DataFrame(absa_data)
        res["keywords"] = self.enrich_aspects_with_news(res["absa_result"])
        res.dropna(inplace=True)
        res["absa_result"] = res["absa_result"].apply(lambda x: x[:10] if len(x) > 10 else x)
        res["keywords"] = res["keywords"].apply(lambda x: x[:5] if len(x) > 5 else x)
        logging.info("Finished ABSA extraction.")
        return res

    def perform_llm_absa(self, group):
        """
        Performs Aspect-Based Sentiment Analysis (ABSA) using the LLM.

        Args:
            group (pd.DataFrame): A DataFrame group corresponding to a specific category.

        Returns:
            dict: A dictionary containing extracted aspects and keywords.
        """
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
        - Then, extract exactly 8 key aspects and their sentiments from the discussions in the table.
        - Aspects should be high-level concepts, not specific entities (e.g., "food quality" instead of "pizza").
        - Do not have duplicate aspects.
        - Aspects should have at least 2 words.
        - There should be exactly 8 aspects.
        - Do not include any additional information.
        - Focus on extracting key aspects and their sentiments.
        - Skip erroneous or missing categories (or where ABSA failed).
        - This is from Singaporean Reddit threads. Please consider the context.
        - Consider all columns in the table when generating the aspects.

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

        Sample Response:
        Car Price, positive
        Vehicle Design, neutral
        Battery Life, negative
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
            if lines:
                for line in lines:
                    if "," in line:
                        aspect_sentiment = line.strip()
                        all_results.append(aspect_sentiment)

            return {"aspects": all_results}
        except Exception as e:
            logging.error(f"ABSA failed: {str(e)}")
            return {"aspects": [f"ABSA failed: {str(e)}"], "keywords": []}

    def chunk_text(self, text, max_tokens):
        """
        Splits a large text into smaller chunks based on the maximum token limit.

        Args:
            text (str): The input text to be chunked.
            max_tokens (int): The maximum number of tokens allowed per chunk.

        Returns:
            list: A list of text chunks.
        """
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
    
    def enrich_aspects_with_news(self, aspects):
        logging.info("Enriching aspects with news headlines.")
        all_headlines = []

        for aspect_pair in aspects:
            aspect = aspect_pair[0].strip()
            headlines = self.newsapi.get_top_headlines(category="sports")
            if headlines:
                all_headlines.extend(headlines)

        if not all_headlines:
            return []

        return self.extract_trending_keywords_from_news(all_headlines)

    def fetch_news_headlines(self, query):
        logging.info(f"Fetching headlines for: {query}")
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "q": query,
            "language": "en",
            "pageSize": 3,
            "apiKey": self.news_api_key
        }

        try:
            response = requests.get(url, params=params)
            articles = response.json().get("articles", [])
            return_val = [a.get("title", "") + " " + a.get("description", "") for a in articles if a.get("title")]
            print(return_val)
            return return_val
        except Exception as e:
            logging.error(f"Failed to fetch news for {query}: {e}")
            return []

    def extract_trending_keywords_from_news(self, headlines):
        logging.info("Extracting trending keywords.")
        headlines_text = "\n".join(headlines[:15])

        prompt = f"""
        You are an expert trend analyst.
        You will be provided with a list of recent news headlines.
        Your task is to extract trending keywords from the headlines.

        Below are recent news headlines across various topics:
        ```
        {headlines_text}
        ```

        **Instructions:**
        - Extract 8 trending keywords from the content above.
        - Keywords can be single words or phrases but must be self-sufficient and informative. 
        - Focus on PROPER noun and associated trends.
        - Do NOT include duplicates.
        - Do NOT explain anything or provide additional text.

        **Output Format:**
        keyword1, keyword2, ..., keyword8

        **Example Output:**
        Tesla stocks, Taylor Swift tickets, Clementi accident, COVID-19
        """
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            return [kw.strip() for kw in result_text.split(",") if kw.strip()]
        except Exception as e:
            logging.error(f"Keyword extraction failed: {str(e)}")
            return []

