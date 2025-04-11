import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from insight_generator.base_decorator import InsightDecorator
import logging

# Configure logging
logging.basicConfig(filename='category_summarizer.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class CategorySummarizerDecorator(InsightDecorator):
    """
    A decorator class for summarizing insights from a DataFrame grouped by categories.
    This class uses a generative AI model to generate summaries, concerns, and suggestions
    for each category in the DataFrame.

    Attributes:
        wrapped (InsightDecorator): The wrapped object to decorate.
        text_col (str): The column containing text data to summarize. If None, it will be inferred.
        category_col (str): The column containing category labels. Defaults to "category".
    """
    def __init__(self, wrapped, text_col=None, category_col="category"):
        """
        Initializes the CategorySummarizerDecorator.

        Args:
            wrapped (InsightDecorator): The wrapped object to decorate.
            text_col (str, optional): The column containing text data to summarize. Defaults to None.
            category_col (str, optional): The column containing category labels. Defaults to "category".

        Raises:
            ValueError: If the GOOGLE_API_KEY is missing in the environment variables.
        """
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
        """
        Extracts insights from the DataFrame by summarizing text data for each category.

        Args:
            df (pd.DataFrame): The input DataFrame containing text and category data.

        Returns:
            pd.DataFrame: A DataFrame containing summaries, concerns, suggestions, and sentiment for each category.

        Raises:
            KeyError: If required text columns ('title' and 'description') are missing.
        """
        logging.info("Starting category summarization.")
        if self.text_col is None:
            if {"title", "description"}.issubset(df.columns):
                df["title_with_desc"] = df["title"].astype(str) + " " + df["description"].astype(str)
                self.text_col = "title_with_desc"
            else:
                raise KeyError("Missing required text columns: title and description")

        summary_data = []
        for category, group in df.groupby(self.category_col):
            logging.info(f"Processing category: {category}")
            try:
                combined_sentiment = group["sentiment"].mean() if "sentiment" in group.columns else None
                summary_result = self.generate_summary(group)  # Pass the DataFrame group
                summary_data.append({
                    self.category_col: category,
                    "summary": summary_result.get("summary", "No summary available"),
                    "concerns": summary_result.get("concerns", []),
                    "suggestions": summary_result.get("suggestions", []),
                    "sentiment": combined_sentiment
                })
            except Exception as e:
                logging.error(f"Error processing category '{category}': {e}")

        res = pd.DataFrame(summary_data)
        res.dropna(inplace=True)
        logging.info("Finished category summarization.")
        return res

    def generate_summary(self, group):
        """
        Generates a summary for a given category using the generative AI model.

        Args:
            group (pd.DataFrame): A DataFrame containing data for a specific category.

        Returns:
            dict: A dictionary containing the summary, concerns, and suggestions.

        Example Output:
            {
                "summary": "The discussions mainly revolve around the impact of the pandemic.",
                "concerns": ["Young adults are facing challenges in finding employment."],
                "suggestions": ["Implement measures to support job creation for young adults."]
            }
        """
        logging.info("Generating summary using LLM")
        dataframe_string = group.to_string(index=False)
        user_prompt = f"""
        You are analyzing Reddit posts related to a specific topic. Your task is to extract meaningful insights.

        **Rules to follow:**
        - DO NOT ask for additional input.
        - DO NOT include the user prompt in the response.
        - DO NOT include the Reddit posts in the response.
        - This is based on the Singapore context so please ensure the output is relevant to the region.
        - Ensure the output is concise and insightful.
        - Focus on summarizing trends, concerns, and suggestions.
        - Suggestions must be actionable and relevant to the context.
        - Consider all columns in the table when generating the summary, concerns, and suggestions.

        - Output must be structured as follows:

        Summary:
        [Brief overview of the discussions - atleast 1 paragraph of 4 lines]

        Concerns:
        - [Summarized concern]
        - [Summarized concern]

        Suggestions:
        - [Summarized suggestion]
        - [Summarized suggestion]

        **Reddit Posts (provided as a table):**
        ```
        {dataframe_string}
        ```

        Generate an insightful summary based on the above content.

        Sample Output:

        Summary:

        The discussions mainly revolve around the impact of the pandemic on the economy and healthcare system. These worries mainly center on the rising prices of essential goods and the mental health of young adults. The posts reflect a growing concern about job security and the need for better healthcare services. The problem of mental health is also highlighted, with many users sharing their personal experiences and seeking support.
        
        Concerns:
        - Young adults are facing challenges in finding employment.
        - The healthcare system is overwhelmed due to the rising number of cases.
        - Mental health issues are on the rise among the population.

        Suggestions:
        - Implement measures to support job creation for young adults.
        - Enhance the healthcare infrastructure to cope with the increasing cases.
        - Provide mental health support services to address the rising concerns.

        MAKE SURE NO CATEGORY IS LEFT OUT! DO NOT RETURN EMPTY INFORMATION!
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
            logging.error(f"Summary generation failed: {str(e)}")
            return {"summary": "Summary generation failed", "concerns": [], "suggestions": []}
