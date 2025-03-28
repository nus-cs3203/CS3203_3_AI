from fastapi import FastAPI, BackgroundTasks
from process_complaints_api import process_complaints, get_task_status, DateRangeRequest as ProcessDateRangeRequest
from category_analytics_api import generate_category_analytics, get_category_analytics_status, DateRangeRequest
from insight_generator.poll_generator_api import generate_poll_prompts, DateRangeRequest as PollDateRangeRequest
import uvicorn

app = FastAPI(title="Complaints Analysis API")

# Mount the complaints processing endpoints
@app.post("/process_complaints")
async def process_complaints_endpoint(request: ProcessDateRangeRequest, background_tasks: BackgroundTasks):
    return await process_complaints(request, background_tasks)

@app.get("/task_status/{task_id}")
async def task_status_endpoint(task_id: str):
    return await get_task_status(task_id)

# Mount the category analytics endpoint
@app.post("/generate_category_analytics")
async def category_analytics_endpoint(request: DateRangeRequest, background_tasks: BackgroundTasks):
    return await generate_category_analytics(request, background_tasks)

@app.get("/category_analytics_status/{task_id}")
async def category_analytics_status_endpoint(task_id: str):
    return await get_category_analytics_status(task_id)

# Mount the poll generator endpoint
@app.post("/generate_poll_prompts")
async def poll_prompts_endpoint(request: PollDateRangeRequest, background_tasks: BackgroundTasks):
    return await generate_poll_prompts(request, background_tasks)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 