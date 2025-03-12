import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator

class CategoryABSAWithLLMInsightDecorator(InsightDecorator):
    def __init__(self, wrapped, text_col=None, category_col="domain_category", max_tokens=3000):
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
        if self.text_col is None:
            if {"title", "description"}.issubset(df.columns):
                df["title_with_desc"] = df["title"].astype(str) + " " + df["description"].astype(str)
                self.text_col = "title_with_desc"
            else:
                raise KeyError("Missing required text columns: title and description")

        absa_data = []
        with open("category_absa_log.txt", "w", encoding="utf-8") as log_file:
            for category, group in df.groupby(self.category_col):
                category_results = []
                keywords = set()
                
                for _, row in group.iterrows():
                    text = row[self.text_col]
                    if pd.notna(text) and text.strip():
                        absa_result = self.perform_llm_absa(text)
                        category_results.extend(absa_result["aspects"])
                        keywords.update(absa_result["keywords"])
                
                absa_data.append({
                    self.category_col: category,
                    "absa_result": category_results,
                    "keywords": list(keywords)
                })
                log_file.write(f"Category: {category}\nABSA: {category_results}\nKeywords: {list(keywords)}\n\n")

        return pd.DataFrame(absa_data)

    def perform_llm_absa(self, text):
        text_chunks = self.chunk_text(text, self.max_tokens)
        all_results = []
        keywords = set()
        
        for chunk in text_chunks:
            user_prompt = f"""
            Extract exactly 5 key aspects and their sentiments from the text.
            
            **Output Format (strictly follow this format):**
            <Aspect>, positive/neutral/negative
            <Aspect>, positive/neutral/negative
            <Aspect>, positive/neutral/negative
            <Aspect>, positive/neutral/negative
            <Aspect>, positive/neutral/negative
            
            **Instructions:**
            - DO NOT ask for additional input.
            - Do not include any additional information.
            - Focus on extracting key aspects and their sentiments
            - Skip erraneous or missing categories (or where ABSA failed)
            - This is from Singaporean Reddit threads. Please consider the context.

            **Text:**
            {chunk}
            """
            try:
                response = self.model.generate_content(user_prompt)
                result_text = response.text.strip() if response and hasattr(response, "text") else ""
                lines = result_text.splitlines()
                for line in lines:
                    if "," in line:
                        aspect_sentiment = line.strip()
                        all_results.append(aspect_sentiment)
                        aspect = aspect_sentiment.split(",")[0].strip()
                        keywords.add(aspect)
            except Exception as e:
                all_results.append(f"ABSA failed: {str(e)}")
        
        return {"aspects": all_results, "keywords": list(keywords)}

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