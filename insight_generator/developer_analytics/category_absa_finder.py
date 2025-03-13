import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator

class CategoryABSAWithLLMInsightDecorator(InsightDecorator):
    def __init__(self, wrapped, text_col=None, category_col="Domain Category", max_tokens=3000):
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
        insights = self._wrapped.extract_insights(df)

        if self.text_col is None:
            if {"title", "selftext"}.issubset(df.columns):
                df["title_selftext"] = df["title"].astype(str) + " " + df["selftext"].astype(str)
                self.text_col = "title_selftext"
            else:
                raise KeyError("Missing required text columns: title and selftext")

        absa_data = []
        with open("category_absa_log.txt", "w", encoding="utf-8") as log_file:
            for category, group in df.groupby(self.category_col):
                combined_text = " ".join(group[self.text_col].dropna().astype(str))
                if combined_text.strip():
                    absa_result = self.perform_llm_absa(combined_text)
                    absa_data.append({
                        self.category_col: category,
                        "aspects": absa_result["aspects"],
                        "sentiments": absa_result["sentiments"],
                        "keywords": absa_result["keywords"]
                    })
                    log_file.write(f"Category: {category}\nABSA: {absa_result}\n\n")

        absa_df = pd.DataFrame(absa_data)
        return df.merge(absa_df, on=self.category_col, how="left")

    def perform_llm_absa(self, text):
        text_chunks = self.chunk_text(text, self.max_tokens)
        all_aspects = []
        all_sentiments = []
        keywords = set()
        
        for chunk in text_chunks:
            user_prompt = f"""
            Extract exactly 5 key aspects and their sentiments from the text.
            
            **Output Format (strictly follow this format):**
            Aspect 1 | positive/neutral/negative
            Aspect 2 | positive/neutral/negative
            Aspect 3 | positive/neutral/negative
            Aspect 4 | positive/neutral/negative
            Aspect 5 | positive/neutral/negative
            
            **Text:**
            {chunk}
            """
            try:
                response = self.model.generate_content(user_prompt)
                result_text = response.text.strip() if response and hasattr(response, "text") else ""
                lines = result_text.splitlines()
                for line in lines:
                    if " | " in line:
                        aspect, sentiment = line.split(" | ")
                        all_aspects.append(aspect.strip())
                        all_sentiments.append(sentiment.strip())
                        keywords.add(aspect.strip())
            except Exception as e:
                all_aspects.append(f"ABSA failed: {str(e)}")
                all_sentiments.append("error")
        
        return {"aspects": all_aspects, "sentiments": all_sentiments, "keywords": list(keywords)}

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
