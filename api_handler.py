from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime

# Import all necessary functions from main_pipeline
from common_components.data_preprocessor.concrete_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.length_validator import LengthValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.validator_logger import ValidatorLogger
from categorizer.deepseek_categorizer_chunked import categorize_complaints
from categorizer.post_process_data import post_process_data
from sentiment_analyser.emotion.distilroberta import DistilRobertaClassifier
from sentiment_analyser.emotion.roberta import RobertaClassifier
from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.polarity.bert import BERTClassifier
from sentiment_analyser.polarity.vader import VaderSentimentClassifier

app = FastAPI()

class Post(BaseModel):
    author_flair_text: str
    created_utc: int
    downs: int
    likes: Optional[int]
    name: str
    no_follow: bool
    num_comments: int
    score: int
    selftext: str
    title: str
    ups: int
    upvote_ratio: float
    url: str
    view_count: Optional[int]
    comments: str

class PostRequest(BaseModel):
    posts: List[Post]

@app.post("/process_complaints")
async def process_complaints(request: PostRequest):
    try:
        # Convert posts to DataFrame
        df = pd.DataFrame(request.dict()["posts"])
        
        # Combine title and selftext
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
            raise ValueError(f"Validation failed: {validation_result['errors']}")
        
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
            print(f"\n===== Running {name} Sentiment Analysis =====")
            context = SentimentAnalysisContext(classifier)
            df = context.analyze(df, text_cols=["title_with_desc"])
        
        print("Columns after sentiment analysis:", df.columns)
        print("Sample row:", df.iloc[0])
        
        # Format response
        complaints = []
        for _, row in df.iterrows():
            # Only include complaints where Intent Category is "Yes"
            if row["Intent Category"] != "Yes":
                continue
            
            complaint = {
                "id": row["name"],
                "title": row["title"],
                "description": row["selftext"],
                "category": row["Domain Category"],
                "date": datetime.fromtimestamp(row["created_utc"]).strftime("%d-%m-%Y %H:%M:%S"),
                "sentiment": float(row["title_with_desc_score"]),
                "url": row["url"],
                "source": "Reddit"
            }
            complaints.append(complaint)
        
        return {"complaints": complaints}
        
    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("Error occurred:", error_detail)  
        raise HTTPException(status_code=500, detail=error_detail)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 