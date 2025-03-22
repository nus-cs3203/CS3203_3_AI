import requests
from datetime import datetime
import asyncio

from category_analytics_api import fetch_complaints, generate_category_analytics

async def test_generate_category_analytics():
    # Define date range
    start_date = "01-05-2024 10:00:00"
    end_date = "01-06-2024 10:59:59"
    
    # Fetch complaints from backend
    # complaints_data = generate_category_analytics(start_date, end_date)
    complaints_data = await generate_category_analytics(start_date, end_date)
    print("\n===== Category Analytics Data =====")
    print(complaints_data["category_analytics"])
    
if __name__ == "__main__":
    asyncio.run(test_generate_category_analytics())
