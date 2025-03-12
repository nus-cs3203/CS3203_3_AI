from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
import requests

# Import all necessary functions from main_pipeline
from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.validator_logger import ValidatorLogger

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator
from insight_generator.category_analytics.llm_category_absa import CategoryABSAWithLLMInsightDecorator
from insight_generator.category_analytics.llm_category_summarizer import CategorySummarizerDecorator

app = FastAPI()

class Complaint(BaseModel):
    title: str
    source: str
    category: str
    date: datetime
    sentiment: float
    description: str
    url: str

class ComplaintRequest(BaseModel):
    complaints: List[Complaint]

@app.post("/generate_category_analytics")
async def generate_category_analytics(request: ComplaintRequest):
    try:
        # Convert posts to DataFrame
        df = pd.DataFrame(request.dict()["complaints"])
        
        # Combine title and selftext
        df["title_with_desc"] = df["title"] + " " + df["description"]
        if "category" in df.columns:
            df.rename(columns={"category": "domain_category"}, inplace=True)

        # Define critical and text columns
        CRITICAL_COLUMNS = ["title_with_desc", "sentiment"]
        TEXT_COLUMNS = ["title_with_desc"]
        
        # Step 1: Preprocessing
        builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, text_columns=TEXT_COLUMNS, data=df, subset=TEXT_COLUMNS)
        director = PreprocessingDirector(builder)
        director.construct_builder()
        df = builder.get_result()
        
        # Step 2: Validation
        logger = ValidatorLogger()
        validator_chain = (
            NotEmptyValidator(CRITICAL_COLUMNS, logger)
            .set_next(OnlyStringValidator(TEXT_COLUMNS, logger))
        )

        validator_chain.validate(df)
        
        # Apply decorators
        base_generator = BaseInsightGenerator()

        # Forecasted insights
        forecast_decorator = TopicSentimentForecastDecorator(base_generator)  # Pass instance, not class
        forecast_insights = forecast_decorator.extract_insights(df)

        # ABSA results
        absa_decorator = CategoryABSAWithLLMInsightDecorator(base_generator)
        absa_insights = absa_decorator.extract_insights(df)

        # Summarized insights
        summary_decorator = CategorySummarizerDecorator(base_generator)
        summary_insights = summary_decorator.extract_insights(df)

        # Combine insights
        insights = absa_insights.merge(forecast_insights, on='domain_category', how='outer').merge(summary_insights, on='domain_category', how='outer')

        # Format response
        categories_analysis = []
        for _, row in insights.iterrows():
            category_analysis = {
                "summary": row.get("summary", ""),
                "keywords": row.get("keywords", []),
                "concerns": row.get("concerns", []),
                "suggestions": row.get("suggestions", []),
                "sentiment": float(row["sentiment"]),
                "forecasted_sentiment": row.get("forecasted_sentiment", 0.0),
                "absa_result": row.get("absa_result", [])
            }
            categories_analysis.append(category_analysis)
        
        return {"category_analytics": categories_analysis}
        
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