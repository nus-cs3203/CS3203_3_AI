from categorizer_api import categorize_for_api
from sentiment_analyser.pipeline_builder import SentimentPipelineBuilder
from sentiment_analyser.pipeline_director import SentimentPipelineDirector
from sentiment_analyser.classifiers.vader_classifier import VaderSentimentClassifier
from datetime import datetime
import pandas as pd
from validators.reddit_post_validator import RedditPostValidator

def process_complaints(request_data):
    """
    Process complaints from posts and return analytics
    
    Args:
        request_data: Dictionary containing posts data in specified format
        {
             "posts": [
                {
                "author_flair_text": "string",
                "created_utc": "integer (unix timestamp)",
                "downs": "integer",
                "likes": "integer or null",
                "name": "string",
                "no_follow": "boolean",
                "num_comments": "integer",
                "score": "integer",
                "selftext": "string",
                "title": "string",
                "ups": "integer",
                "upvote_ratio": "float",
                "url": "string (URL)",
                "view_count": "integer or null",	
                "comments": "string"
                }
  ]

        }
    
    Returns:
        Dictionary containing processed complaints in specified format
        {
            "complaints": [
                {
                    "id": string,
                    "title": string,
                    "description": string,
                    "category": string,
                    "date": string,
                    "sentiment": float,
                    "url": string,
                    "source": string
                }
            ]
        }
    """
    # Validate request data
    validator = RedditPostValidator()
    validation_result = validator.validate_request(request_data)
    if "error" in validation_result:
        return {"error": validation_result["error"]}

    posts = request_data.get("posts", [])
    complaints = []

    # Convert posts to DataFrame for categorization
    df = pd.DataFrame(posts)

    # Get categories using new function
    categories = categorize_for_api(df)

    for i, post in enumerate(posts):
        # Create complaint ID using post name
        complaint_id = post['name']  # Already unique for Reddit posts

        # Get sentiment
        text = f"{post['title']} {post['selftext']}"
        sentiment_pipeline = SentimentPipelineBuilder(
            strategy=VaderSentimentClassifier()  
        )
        director = SentimentPipelineDirector(sentiment_pipeline)
        sentiment_result = director.construct_pipeline(text)
        
        # Format date according to specified format
        date_str = datetime.fromtimestamp(post['created_utc']).strftime("%d-%m-%Y %H:%M:%S")
        
        # Create complaint object according to specified format
        complaint = {
            "id": complaint_id,
            "title": post['title'],
            "description": post['selftext'],
            "intent_category": categories.iloc[i]['Intent Category'],
            "domain_category": categories.iloc[i]['Domain Category'],
            "date": date_str,
            "sentiment": float(sentiment_result['score']),
            "url": post['url'],
            "source": "Reddit"
        }
        
        complaints.append(complaint)

    return {"complaints": complaints}

# # Example usage for testing
# if __name__ == "__main__":
#     test_data = {
#         "posts": [
#             {
#                 "author_flair_text": "Resident",
#                 "created_utc": 1677649200,
#                 "downs": 2,
#                 "likes": None,
#                 "name": "abc123",
#                 "no_follow": True,
#                 "num_comments": 5,
#                 "score": 8,
#                 "selftext": "The education system in Singapore is so bad",
#                 "title": "I dont like the education system in Singapore",
#                 "ups": 10,
#                 "upvote_ratio": 0.8,
#                 "url": "https://reddit.com/post1",
#                 "view_count": None,
#                 "comments": "This is bad"
#             },
#             {
#                 "author_flair_text": "Student",
#                 "created_utc": 1677649300,
#                 "downs": 1,
#                 "likes": None,
#                 "name": "def456",
#                 "no_follow": True,
#                 "num_comments": 3,
#                 "score": 15,
#                 "selftext": "The traffic during peak hours is terrible. We need more buses.",
#                 "title": "Public transport needs improvement",
#                 "ups": 16,
#                 "upvote_ratio": 0.9,
#                 "url": "https://reddit.com/post2",
#                 "view_count": None,
#                 "comments": "Agree with this"
#             },
#             {
#                 "author_flair_text": "Worker",
#                 "created_utc": 1677649400,
#                 "downs": 0,
#                 "likes": None,
#                 "name": "ghi789",
#                 "no_follow": True,
#                 "num_comments": 8,
#                 "score": 20,
#                 "selftext": "Housing prices are getting out of control",
#                 "title": "HDB prices are too high",
#                 "ups": 20,
#                 "upvote_ratio": 1.0,
#                 "url": "https://reddit.com/post3",
#                 "view_count": None,
#                 "comments": "Need to do something about this"
#             }
#         ]
#     }
    
#     result = process_complaints(test_data)
#     print("Test result:", result) 