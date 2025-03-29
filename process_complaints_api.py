from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
from typing import List, Dict, Any
import pandas as pd
import requests
from datetime import datetime
import os
import uuid
import json
from pathlib import Path

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

# Store for task results (In a production environment, this should be a database)
TASK_RESULTS = {}
TASKS_DIR = Path("task_results")
TASKS_DIR.mkdir(exist_ok=True)

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
    
    print("API Response:", response.json())  # Debug line to print the entire response
    # Ensure backend response has a 'documents' key
    return response.json()["documents"]

async def process_complaints_background(start_date: str, end_date: str, task_id: str):
    """Process complaints in the background."""
    try:
        # Fetch complaints from backend
        complaints = fetch_complaints(start_date, end_date)
        if not complaints:
            TASK_RESULTS[task_id] = {
                "status": "completed",
                "result": {"message": "No complaints found for the given date range."}
            }
            return

        # Convert to DataFrame
        df = pd.DataFrame(complaints)
        df["title_with_desc"] = df["title"] + " " + df["selftext"]

        # Define critical and text columns
        CRITICAL_COLUMNS = ["title_with_desc"]
        TEXT_COLUMNS = ["title_with_desc", "comments"]

        # Debug line to print the DataFrame after fetching data
        print("DataFrame after fetching data:", df)

        # Preprocessing
        builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, text_columns=TEXT_COLUMNS, data=df)
        director = PreprocessingDirector(builder)
        director.construct_builder()
        df = builder.get_result()

        # Debug line to print the DataFrame after preprocessing
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

        # Debug line to print the DataFrame after validation
        print("DataFrame after validation:", df)

        # Categorization
        categories = [
            "Housing", "Healthcare", "Public Safety", "Transport",
            "Education", "Environment", "Employment", "Public Health",
            "Legal", "Economy", "Politics", "Technology",
            "Infrastructure", "Others"
        ]
        
        df = categorize_complaints(df=df, categories=categories)

        # Debug line to print the DataFrame after categorization
        print("DataFrame after categorization:", df)

        # Sentiment Analysis
        classifiers = [
            # ("BERT", BERTClassifier()),
            ("VADER", VaderSentimentClassifier())
            # ("DistilRoberta Emotion", DistilRobertaClassifier()),
            # ("Roberta Emotion", RobertaClassifier()),
        ]

        for name, classifier in classifiers:
            print(f"\n===== Running {name} Sentiment Analysis =====")
            context = SentimentAnalysisContext(classifier)
            df = context.analyze(df, text_cols=["title_with_desc"])

        # Debug line to print the DataFrame after sentiment analysis
        print("DataFrame after sentiment analysis:", df)
        
        # Post-processing
        df = post_process_data(df=df)

        # Debug line to print the DataFrame after post-processing
        print("DataFrame after post-processing:", df)

        # Convert the DataFrame to a list of dictionaries
        data = df.to_dict(orient="records")
        
        # Save result with data
        with open(TASKS_DIR / f"{task_id}.json", "w") as f:
            json.dump({
                "status": "completed", 
                "result": {
                    "message": "Sentiment analysis completed successfully.",
                    "data": data
                }
            }, f)
        
    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("Error occurred:", error_detail)
        TASK_RESULTS[task_id] = {
            "status": "failed",
            "error": error_detail
        }

@app.post("/process_complaints")
async def process_complaints(request: DateRangeRequest, background_tasks: BackgroundTasks):
    """
    Start the sentiment analysis process in the background.
    Returns a task ID that can be used to check the status.
    """
    task_id = str(uuid.uuid4())
    TASK_RESULTS[task_id] = {"status": "processing"}
    
    # Add task to background tasks
    background_tasks.add_task(process_complaints_background, request.start_date, request.end_date, task_id)
    
    return {
        "message": "Processing started. You can check the status using the task_id.",
        "task_id": task_id
    }

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a processing task."""
    task_file = TASKS_DIR / f"{task_id}.json"

    # ✅ Check local file first (Heroku dynos may restart, in-memory data could be lost)
    if task_file.exists():
        with open(task_file, "r") as f:
            file_result = json.load(f)
        return file_result.get("result", {"status": file_result.get("status", "unknown")})

    # If not found in file, check in-memory storage (normal case when no restart)
    if task_id in TASK_RESULTS:
        task_result = TASK_RESULTS[task_id]

        if task_result["status"] == "completed":
            return task_result["result"]
        elif task_result["status"] == "failed":
            raise HTTPException(status_code=500, detail=task_result["error"])
        else:
            return {"status": "processing"}

    # Finally, if not found anywhere
    raise HTTPException(status_code=404, detail="Task not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)