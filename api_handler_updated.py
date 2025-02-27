from categorizer_api import categorize_for_api
from datetime import datetime
import pandas as pd

from common_components.data_preprocessor.concrete_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from common_components.data_preprocessor.concrete_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.length_validator import LengthValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.validator_logger import ValidatorLogger
from categorizer.deepseek_categorizer_chunked import categorize_complaints
from categorizer.post_process_data import post_process_data
from sentiment_analyser.classifiers.polarity.bert import BERTClassifier
from sentiment_analyser.classifiers.polarity.vader import VaderSentimentClassifier
from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.emotion.distilroberta import DistilRobertaClassifier
from sentiment_analyser.emotion.roberta import RobertaClassifier

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
                    "intent_category": string,
                    "domain_category": string,
                    "date": string,
                    "sentiment": float,
                    "url": string,
                    "source": string
                }
            ]
        }
    """

    posts = request_data.get("posts", [])
    complaints = []

    # Convert posts to DataFrame for categorization
    df = pd.DataFrame(posts)
    df.dropna(subset=["title", "selftext"], inplace=True)
    df["title_with_desc"] = df["title"] + " " + df["selftext"]

    # Define critical and text columns
    CRITICAL_COLUMNS = ["title_with_desc"]
    TEXT_COLUMNS = ["title_with_desc", "comments"]

    # Step 1: Preprocessing
    builder = GeneralPreprocessorBuilder(CRITICAL_COLUMNS, TEXT_COLUMNS)
    director = PreprocessingDirector(builder)
    df = director.construct(df)

    # Step 2: Validation
    logger = ValidatorLogger()
    validator_chain = (
        NotEmptyValidator(CRITICAL_COLUMNS, logger)
        .set_next(OnlyStringValidator(TEXT_COLUMNS, logger))
        .set_next(LengthValidator({"title_with_desc": (5, 100)}, logger))
    )

    validation_result = validator_chain.validate(df)
    if not validation_result["success"]:
        print("Validation failed:", validation_result["errors"])
        exit(1)

    # Step 3: Categorization & Post-processing
    df = categorize_complaints(df=df)
    df = post_process_data(df=df)

    # Step 4: Sentiment Analysis
    classifiers = [
        ("BERT", BERTClassifier()),
        ("VADER", VaderSentimentClassifier()),
        ("DistilRoberta Emotion", DistilRobertaClassifier()),
        ("Roberta Emotion", RobertaClassifier()),
    ]

    for name, classifier in classifiers:
        context = SentimentAnalysisContext(classifier)
        df = context.analyze(df, text_cols=["title_with_desc"])

    return df.to_json(orient="records")

# # Example usage for testing
if __name__ == "__main__":
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
                "selftext": "The education system in Singapore is so bad",
                "title": "I dont like the education system in Singapore",
                "ups": 10,
                "upvote_ratio": 0.8,
                "url": "https://reddit.com/post1",
                "view_count": None,
                "comments": "This is bad"
            },
            {
                "author_flair_text": "Student",
                "created_utc": 1677649300,
                "downs": 1,
                "likes": None,
                "name": "def456",
                "no_follow": True,
                "num_comments": 3,
                "score": 15,
                "selftext": "The traffic during peak hours is terrible. We need more buses.",
                "title": "Public transport needs improvement",
                "ups": 16,
                "upvote_ratio": 0.9,
                "url": "https://reddit.com/post2",
                "view_count": None,
                "comments": "Agree with this"
            },
            {
                "author_flair_text": "Worker",
                "created_utc": 1677649400,
                "downs": 0,
                "likes": None,
                "name": "ghi789",
                "no_follow": True,
                "num_comments": 8,
                "score": 20,
                "selftext": "Housing prices are getting out of control",
                "title": "HDB prices are too high",
                "ups": 20,
                "upvote_ratio": 1.0,
                "url": "https://reddit.com/post3",
                "view_count": None,
                "comments": "Need to do something about this"
            }
        ]
    }
    
    result = process_complaints(test_data)
    print("Test result:", result) 