from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import pandas as pd
import requests
from datetime import datetime

# Import necessary functions
from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.validator_logger import ValidatorLogger
from categorizer.post_process_data import post_process_data
from categorizer.r1_categorizer import categorize_complaints  # Ensure this import is correct
from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.emotion.distilroberta import DistilRobertaClassifier
from sentiment_analyser.emotion.roberta import RobertaClassifier
from sentiment_analyser.polarity.bert import BERTClassifier
from sentiment_analyser.polarity.vader import VaderSentimentClassifier

app = FastAPI()

class DateRangeRequest(BaseModel):
    start_date: str  # Format: "dd-mm-yyyy HH:MM:SS"
    end_date: str    # Format: "dd-mm-yyyy HH:MM:SS"

    @validator("start_date", "end_date")
    def validate_date_format(cls, value):
        try:
            datetime.strptime(value, "%d-%m-%Y %H:%M:%S")
        except ValueError:
            raise ValueError("Incorrect date format, should be 'dd-mm-yyyy HH:MM:SS'")
        return value

def fetch_complaints(start_date: str, end_date: str):
    """Fetch complaints from backend API."""
    url = "http://localhost:8003/complaints/get_by_daterange"
    headers = {"Content-Type": "application/json"}
    payload = {"start_date": start_date, "end_date": end_date}
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch complaints")
    
    return response.json()["complaints"]  # Ensure backend response has a 'complaints' key

@app.post("/perform_sentiment_analysis")
async def perform_sentiment_analysis(request: DateRangeRequest):
    try:
        # Fetch complaints from backend
        complaints = fetch_complaints(request.start_date, request.end_date)
        if not complaints:
            return {"message": "No complaints found for the given date range."}

        # Convert to DataFrame
        df = pd.DataFrame(complaints)
        df["title_with_desc"] = df["title"] + " " + df["description"]

        # Define critical and text columns
        CRITICAL_COLUMNS = ["title_with_desc"]
        TEXT_COLUMNS = ["title_with_desc", "comments"]

        # Preprocessing
        builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, text_columns=TEXT_COLUMNS, data=df)
        director = PreprocessingDirector(builder)
        director.construct_builder()
        df = builder.get_result()

        # Validation
        logger = ValidatorLogger()
        validator_chain = (
            NotEmptyValidator(CRITICAL_COLUMNS, logger)
            .set_next(OnlyStringValidator(TEXT_COLUMNS, logger))
        )
        validation_result = validator_chain.validate(df)
        if not validation_result["success"]:
            raise ValueError(f"Validation failed: {validation_result['errors']}")

        # Categorization
        categories = [
            "Housing", "Healthcare", "Public Safety", "Transport",
            "Education", "Environment", "Employment", "Public Health",
            "Legal", "Economy", "Politics", "Technology",
            "Infrastructure", "Others"
        ]
        df = categorize_complaints(df=df, categories=categories)

        # Sentiment Analysis
        classifiers = [
            ("BERT", BERTClassifier()),
            ("VADER", VaderSentimentClassifier()),
            ("DistilRoberta Emotion", DistilRobertaClassifier()),
            ("Roberta Emotion", RobertaClassifier()),
        ]

        for name, classifier in classifiers:
            print(f"\n===== Running {name} Sentiment Analysis =====")
            context = SentimentAnalysisContext(classifier)
            df = context.analyze(df, text_cols=["title_with_desc"])

        # Post-processing
        df = post_process_data(df=df)

        # Ensure the output matches the schema
        df['id'] = df['name']
        df['category'] = df['Domain Category']
        df['date'] = df['created_utc'].apply(lambda x: datetime.fromtimestamp(x).strftime("%d-%m-%Y %H:%M:%S"))
        df['sentiment'] = df['title_with_desc_score']
        df['source'] = "Reddit"

        # Select and reorder columns to match the schema
        output_df = df[['id', 'title', 'description', 'category', 'date', 'sentiment', 'url', 'source']]

        # Return the processed DataFrame as CSV
        output_csv = "csv_results/sentiment_analysis_result.csv"
        output_df.to_csv(output_csv, index=False)
        return {"message": "Sentiment analysis completed successfully.", "csv_path": output_csv}

    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("Error occurred:", error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)