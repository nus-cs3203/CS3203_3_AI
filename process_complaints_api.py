from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import pandas as pd
import requests
from datetime import datetime
import os

# Import necessary functions
from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.validator_logger import ValidatorLogger
from categorizer.post_process_data import post_process_data
from categorizer.r1_categorizer import categorize_complaints  
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
    url = "https://cs32033backend-management-production.up.railway.app/posts/get_by_daterange"
    headers = {"Content-Type": "application/json"}
    payload = {"start_date": start_date, "end_date": end_date}
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch complaints")
    
    print("API Response:", response.json())  # Debugging line to print the entire response
    # Ensure backend response has a 'documents' key
    return response.json()["documents"]

@app.post("/process_complaints")
async def process_complaints(request: DateRangeRequest):
    try:
        # Fetch complaints from backend
        complaints = fetch_complaints(request.start_date, request.end_date)
        if not complaints:
            return {"message": "No complaints found for the given date range."}

        # Convert to DataFrame
        df = pd.DataFrame(complaints)
        df["title_with_desc"] = df["title"] + " " + df["selftext"]

        # Define critical and text columns
        CRITICAL_COLUMNS = ["title_with_desc"]
        TEXT_COLUMNS = ["title_with_desc", "comments"]

        # Debugging line to print the DataFrame after fetching data
        print("DataFrame after fetching data:", df)

        # Preprocessing
        builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, text_columns=TEXT_COLUMNS, data=df)
        director = PreprocessingDirector(builder)
        director.construct_builder()
        df = builder.get_result()

        # Debugging line to print the DataFrame after preprocessing
        print("DataFrame after preprocessing:", df)

        # Validation
        logger = ValidatorLogger()
        validator_chain = (
            NotEmptyValidator(CRITICAL_COLUMNS, logger)
            .set_next(OnlyStringValidator(TEXT_COLUMNS, logger))
        )
        validation_result = validator_chain.validate(df)
        if not validation_result["success"]:
            raise ValueError(f"Validation failed: {validation_result['errors']}")

        # Debugging line to print the DataFrame after validation
        print("DataFrame after validation:", df)

        # Categorization
        categories = [
            "Housing", "Healthcare", "Public Safety", "Transport",
            "Education", "Environment", "Employment", "Public Health",
            "Legal", "Economy", "Politics", "Technology",
            "Infrastructure", "Others"
        ]
        
        df = categorize_complaints(df=df, categories=categories)

        # Debugging line to print the DataFrame after categorization
        print("DataFrame after categorization:", df)

        # Sentiment Analysis
        classifiers = [
            # ("BERT", BERTClassifier()),
            ("VADER", VaderSentimentClassifier()),
            # ("DistilRoberta Emotion", DistilRobertaClassifier()),
            # ("Roberta Emotion", RobertaClassifier()),
        ]

        for name, classifier in classifiers:
            print(f"\n===== Running {name} Sentiment Analysis =====")
            context = SentimentAnalysisContext(classifier)
            df = context.analyze(df, text_cols=["title_with_desc"])

        # Debugging line to print the DataFrame after sentiment analysis
        print("DataFrame after sentiment analysis:", df)
        output_csv = "csv_results/sentiment_analysis_result_before_post_processing.csv"
        df.to_csv(output_csv, index=False)
        # Post-processing
        df = post_process_data(df=df)

        # Debugging line to print the DataFrame after post-processing
        print("DataFrame after post-processing:", df)

        # Return the processed DataFrame as CSV
        output_csv = "csv_results/sentiment_analysis_result.csv"
        df.to_csv(output_csv, index=False)
        return {"message": "Sentiment analysis completed successfully.", "csv_path": output_csv}

    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("Error occurred:", error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)