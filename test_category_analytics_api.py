import requests
from datetime import datetime

# Define test data with Reddit posts formatted as Complaint objects
test_data = {
    "complaints": [
        {
            "title": "Education system issues",
            "source": "Reddit",
            "category": "Education",
            "date": datetime.utcfromtimestamp(1677649200).isoformat(),
            "sentiment": 8,  # Use 'score' as sentiment
            "description": "The education system needs improvement",
            "url": "https://reddit.com/post1"
        },
        {
            "title": "Enjoying the weather",
            "source": "Reddit",
            "category": "Environment",
            "date": datetime.utcfromtimestamp(1677649300).isoformat(),
            "sentiment": 15,  # Use 'score' as sentiment
            "description": "Today is such a beautiful day",
            "url": "https://reddit.com/post2"
        }
    ]
}

# Send the test data to the FastAPI endpoint
response = requests.post("http://localhost:8000/generate_category_analytics", json=test_data)

# Print the response
print("Status Code:", response.status_code)
print("Response:", response.json())
if response.status_code != 200:
    print("Error Details:", response.json())