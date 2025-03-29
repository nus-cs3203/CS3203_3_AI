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
import io
import tempfile

# Import the main pipeline function
from main_pipeline import process_pipeline

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
    
    #print("API Response:", response.json())  # Debug line to print the entire response
    # Ensure backend response has a 'documents' key
    return response.json()["documents"]

async def process_complaints_background(start_date: str, end_date: str, task_id: str):
    """Process complaints in the background."""
    try:
        print("\n" + "="*50)
        print(f"🚀 Starting to process complaints for task {task_id}")
        print(f"📅 Date range: {start_date} to {end_date}")
        print("="*50 + "\n")
        
        # Fetch complaints from backend
        complaints = fetch_complaints(start_date, end_date)
        if not complaints:
            TASK_RESULTS[task_id] = {
                "status": "completed",
                "result": {"message": "No complaints found for the given date range."}
            }
            print("\n" + "="*50)
            print(f"⚠️ No complaints found for task {task_id}")
            print("="*50 + "\n")
            return

        # Convert to DataFrame
        df = pd.DataFrame(complaints)
        print(f"📊 Found {len(df)} complaints to process")
        
        # Create temporary file and directory for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_input:
            temp_input_path = temp_input.name
            df.to_csv(temp_input_path, index=False)
        
        # Create task-specific output directory
        output_folder = TASKS_DIR / f"pipeline_results_{task_id}"
        
        # Run the main pipeline
        df_final = process_pipeline(temp_input_path, output_folder)
        
        # Convert the results to a list of dictionaries for the API response
        if df_final is not None and not df_final.empty:
            data = df_final.to_dict(orient="records")
            print("\n" + "="*50)
            print(f"✨ Processing completed successfully for task {task_id}!")
            print(f"📝 Processed {len(data)} complaints")
            print("="*50 + "\n")
        else:
            data = []
            print("\n" + "="*50)
            print(f"⚠️ No complaints were processed for task {task_id}")
            print("="*50 + "\n")
        
        # Save result with data
        with open(TASKS_DIR / f"{task_id}.json", "w") as f:
            json.dump({
                "status": "completed", 
                "result": {
                    "message": "Sentiment analysis completed successfully.",
                    "data": data
                }
            }, f)
            
        # Clean up temporary file
        os.unlink(temp_input_path)
        
    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("\n" + "="*50)
        print(f"❌ Processing failed for task {task_id}")
        print(f"🔥 Error: {str(e)}")
        print("="*50 + "\n")
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
    
    print(f"Starting complaint processing for date range: {request.start_date} to {request.end_date}")
    print(f"Task ID: {task_id}")
    
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