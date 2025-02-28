import requests

test_data = {
    "posts": [
        {
            "author_flair_text": "Resident",
            "created_utc": 1677649200,
            "downs": 2,
            "likes": None,
            "name": "abc123",
            "no_follow": True,
            "num_comments": 5,
            "score": 8,
            "selftext": "The education system needs improvement",
            "title": "Education system issues",
            "ups": 10,
            "upvote_ratio": 0.8,
            "url": "https://reddit.com/post1",
            "view_count": None,
            "comments": "This needs attention"
        },
        {
            "author_flair_text": "User",
            "created_utc": 1677649300,
            "downs": 0,
            "likes": None,
            "name": "xyz789",
            "no_follow": True,
            "num_comments": 3,
            "score": 15,
            "selftext": "Today is such a beautiful day",
            "title": "Enjoying the weather",
            "ups": 15,
            "upvote_ratio": 0.95,
            "url": "https://reddit.com/post2",
            "view_count": None,
            "comments": "Nice weather indeed"
        }
    ]
}

response = requests.post("http://localhost:8000/process_complaints", json=test_data)

print("Status Code:", response.status_code)
print("Response:", response.json())
if response.status_code != 200:
    print("Error Details:", response.json()) 