from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
from typing import List
import requests
from datetime import datetime
import os
from insight_generator.poll_generator import PollGenerator
import pandas as pd
import uuid
import json
from pathlib import Path
import traceback

app = FastAPI()

# Share the same task storage
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
    url = "https://cs32033backend-management-production.up.railway.app/complaints/get_by_daterange"
    headers = {"Content-Type": "application/json"}
    payload = {"start_date": start_date, "end_date": end_date}
    print("Waiting for response")
    response = requests.post(url, json=payload, headers=headers)
    #print(response)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch complaints")
    
    response_data = response.json()
    if not response_data.get("success", False):
        raise HTTPException(status_code=500, detail=f"Backend request failed: {response_data.get('message', 'Unknown error')}")
    
    print(response_data["documents"])
    return response_data["documents"]

async def generate_poll_prompts_background(start_date: str, end_date: str, task_id: str):
    """Process poll generation in the background."""
    try:
        complaints_data = fetch_complaints(start_date, end_date)
        df = pd.DataFrame(complaints_data)
        
        df = df.rename(columns={'domain_category': 'category'})
        
        print(f"\nStarting to generate poll prompts for task {task_id}...")
        prompt_generator = PollGenerator()
        insights = prompt_generator.extract_insights(df)
        
        result = {
            "message": "Poll prompts generated successfully.",
            "data": insights.to_dict(orient="records")
        }
        
        # Save results both in memory and file
        TASK_RESULTS[task_id] = {
            "status": "completed",
            "result": result
        }
        
        with open(TASKS_DIR / f"{task_id}.json", "w") as f:
            json.dump({"status": "completed", "result": result}, f)
        
        print(f"\n✨ Poll generation completed successfully for task {task_id}!")
        print(f"Generated {len(insights)} poll prompts across different categories.")
            
    except Exception as e:
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        TASK_RESULTS[task_id] = {
            "status": "failed",
            "error": error_detail
        }
        print(f"\n❌ Poll generation failed for task {task_id}")
        print(f"Error: {str(e)}")

@app.post("/generate_poll_prompts")
async def generate_poll_prompts(request: DateRangeRequest, background_tasks: BackgroundTasks):
    """
    Start the poll generation process in the background.
    Returns a task ID that can be used to check the status.
    """
    task_id = str(uuid.uuid4())
    TASK_RESULTS[task_id] = {"status": "processing"}
    
    print("Starting poll prompt generation...")
    print(f"Start date: {request.start_date}, End date: {request.end_date}")
    print(f"Task ID: {task_id}")

    background_tasks.add_task(
        generate_poll_prompts_background,
        request.start_date,
        request.end_date,
        task_id
    )
    
    print("Poll prompt generation task added to background.")
    return {
        "message": "Poll generation started. You can check the status using the task_id.",
        "task_id": task_id
    }

@app.get("/poll_generation_status/{task_id}")
async def poll_generation_status(task_id: str):
    """Get the status of a poll generation task."""
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
    uvicorn.run("insight_generator.poll_generator_api:app", host="127.0.0.1", port=8000, reload=True) 