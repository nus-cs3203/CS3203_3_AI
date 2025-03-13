import requests
from datetime import datetime

def fetch_complaints(start_date: str, end_date: str):
    url = "http://localhost:8083/complaints/get_by_daterange"
    payload = {
        "start_date": start_date,
        "end_date": end_date
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        print("Response message:", data.get("message", ""))
        print("Success:", data.get("success", False))
        return data.get("complaints", [])  # Extract complaints list
    else:
        print("Error fetching complaints:", response.status_code, response.text)
        return None

def test_generate_category_analytics():
    # Define date range
    start_date = "01-01-2022 00:00:00"
    end_date = "12-12-2023 23:59:59"
    
    # Fetch complaints from backend
    complaints_data = fetch_complaints(start_date, end_date)
    if not complaints_data:
        print("No complaints data retrieved.")
        return
    
    # Send the fetched data to the FastAPI endpoint
    url = "http://localhost:8000/generate_category_analytics"
    response = requests.post(url, json={"complaints": complaints_data})
    
    # Print the response
    print("Status Code:", response.status_code)
    try:
        response_json = response.json()
        print("Response:", response_json)
        if response.status_code != 200:
            print("Error Details:", response_json)
    except Exception as e:
        print("Failed to parse response as JSON:", str(e))

if __name__ == "__main__":
    test_generate_category_analytics()