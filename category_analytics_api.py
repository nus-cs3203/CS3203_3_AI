from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
from typing import List
import pandas as pd
import requests
from datetime import datetime
import uuid
import json
from pathlib import Path

# Import all necessary functions from main_pipeline
from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.validator_logger import ValidatorLogger

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator
from insight_generator.category_analytics.llm_category_absa import CategoryABSAWithLLMInsightDecorator
from insight_generator.category_analytics.llm_category_summarizer import CategorySummarizerDecorator

app = FastAPI()

# Add storage for task results
TASK_RESULTS = {}
TASKS_DIR = Path("category_analytics_results")
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
    try:
        datetime.strptime(start_date, "%d-%m-%Y %H:%M:%S")
        datetime.strptime(end_date, "%d-%m-%Y %H:%M:%S")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format, should be 'dd-mm-yyyy HH:MM:SS'")

    url = "https://cs32033backend-management-production.up.railway.app/complaints/get_by_daterange"
    headers = {"Content-Type": "application/json"}
    payload = {"start_date": start_date, "end_date": end_date}
    print("Waiting for response")
    response = requests.post(url, json=payload, headers=headers)
    # print(response)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch complaints")
    
    response_data = response.json()
    if not response_data.get("success", False):
        raise HTTPException(status_code=500, detail=f"Backend request failed: {response_data.get('message', 'Unknown error')}")
    
    print(response_data["documents"])
    return response_data["documents"]

async def generate_category_analytics_background(start_date: str, end_date: str, task_id: str):
    """Process category analytics in the background."""
    try:
        # Fetch complaints from backend
        complaints = fetch_complaints(start_date, end_date)
        if not complaints:
            TASK_RESULTS[task_id] = {
                "status": "completed",
                "result": {"category_analytics": []}
            }
            return

        # Convert to DataFrame
        df = pd.DataFrame(complaints)
        # Adjust to match the new data structure

        if "domain_category" in df.columns:
            df.rename(columns={"domain_category": "category"}, inplace=True)
        
        # Check if title and description columns are present
        if "selftext" in df.columns:
            df["description"] = df["selftext"]
        if "title" in df.columns and "description" in df.columns:
            df["title_with_desc"] = df["title"] + " " + df["description"]
        elif "title" in df.columns:
            df["title_with_desc"] = df["title"]
        elif "description" in df.columns:
            df["title_with_desc"] = df["description"]
        else:
            raise HTTPException(status_code=400, detail="No title or description found in the data.")
        
        # Define critical and text columns
        CRITICAL_COLUMNS = ["title_with_desc"]
        TEXT_COLUMNS = ["title_with_desc"]

        # Preprocessing
        builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, text_columns=TEXT_COLUMNS, data=df, subset=TEXT_COLUMNS)
        director = PreprocessingDirector(builder)
        director.construct_builder()
        df = builder.get_result()

        # Validation
        logger = ValidatorLogger()
        validator_chain = (
            NotEmptyValidator(CRITICAL_COLUMNS, logger)
            .set_next(OnlyStringValidator(TEXT_COLUMNS, logger))
        )
        validator_chain.validate(df)

        # Apply insight decorators
        base_generator = BaseInsightGenerator()
        forecast_insights = TopicSentimentForecastDecorator(base_generator).extract_insights(df)
        absa_insights = CategoryABSAWithLLMInsightDecorator(base_generator).extract_insights(df)
        summary_insights = CategorySummarizerDecorator(base_generator).extract_insights(df)

        # Merge insights
        insights = absa_insights.merge(forecast_insights, on='category', how='outer').merge(summary_insights, on='category', how='outer')

        # Format response
        categories_analysis = []
        for _, row in insights.iterrows():
            category_analysis = {
                "summary": row.get("summary", ""),
                "keywords": row.get("keywords", []),
                "concerns": row.get("concerns", []),
                "suggestions": row.get("suggestions", []),
                "sentiment": float(row["sentiment"]),
                "forecasted_sentiment": row.get("forecasted_sentiment", 0.0),
                "absa_result": row.get("absa_result", [])
            }
            categories_analysis.append(category_analysis)

        # Save results to file
        with open(TASKS_DIR / f"{task_id}.json", "w") as f:
            json.dump({
                "status": "completed",
                "result": {"category_analytics": categories_analysis}
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

@app.post("/generate_category_analytics")
async def generate_category_analytics(request: DateRangeRequest, background_tasks: BackgroundTasks):
    """
    Start the category analytics process in the background.
    Returns a task ID that can be used to check the status.
    """
    task_id = str(uuid.uuid4())
    TASK_RESULTS[task_id] = {"status": "processing"}
    
    # Add task to background tasks
    background_tasks.add_task(
        generate_category_analytics_background, 
        request.start_date, 
        request.end_date, 
        task_id
    )
    
    return {
        "message": "Processing started. You can check the status using the task_id.",
        "task_id": task_id
    }

@app.get("/category_analytics_status/{task_id}")
async def get_category_analytics_status(task_id: str):
    """Get the status of a category analytics processing task."""
    task_file = TASKS_DIR / f"{task_id}.json"

    # Check local file first
    if task_file.exists():
        with open(task_file, "r") as f:
            file_result = json.load(f)
        return file_result.get("result", {"status": file_result.get("status", "unknown")})

    # Check in-memory storage
    if task_id in TASK_RESULTS:
        task_result = TASK_RESULTS[task_id]

        if task_result["status"] == "completed":
            return task_result["result"]
        elif task_result["status"] == "failed":
            raise HTTPException(status_code=500, detail=task_result["error"])
        else:
            return {"status": "processing"}

    raise HTTPException(status_code=404, detail="Task not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
