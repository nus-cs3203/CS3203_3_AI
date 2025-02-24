import pandas as pd
import json

# Import validation components
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.length_validator import LengthValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.text_validator.regex_validator import RegexValidator
from common_components.data_validator.validator_logger import ValidatorLogger

# Import categorization components
from categorizer.deepseek_categorizer_chunked import categorize_complaints
from categorizer.post_process_data import post_process_data

# Import sentiment analysis components
from sentiment_analyser.classifiers.polarity.bert import BERTClassifier
from sentiment_analyser.classifiers.polarity.vader import VaderSentimentClassifier
from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.emotion.distilroberta import DistilRobertaClassifier
from sentiment_analyser.emotion.roberta import RobertaClassifier

# Load config
with open("basic_config.json", "r") as f:
    config = json.load(f)

def preprocess_validate_categorize_analyze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: Preprocess -> Validate -> Categorize -> Sentiment Analyze
    """
    # Step 1: Preprocess
    preprocessing_cfg = config.get("preprocessing", {})
    if preprocessing_cfg.get("remove_missing_values", True):
        df.dropna(subset=preprocessing_cfg.get("critical_columns", ["title", "selftext"]), inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["title_with_desc"] = df["title"] + " " + df["selftext"]
    
    # Step 2: Validate
    logger = ValidatorLogger()
    validation_cfg = config.get("validation", {})
    
    validator_chain = NotEmptyValidator(
        validation_cfg.get("not_empty_columns", ["title", "selftext"]), logger
    )

    if "only_string_columns" in validation_cfg:
        validator_chain.set_next(OnlyStringValidator(validation_cfg["only_string_columns"], logger))

    if "length_constraints" in validation_cfg:
        validator_chain.set_next(LengthValidator(validation_cfg["length_constraints"], logger))

    if "regex_constraints" in validation_cfg:
        regex_cols = list(validation_cfg["regex_constraints"].keys())
        regex_patterns = list(validation_cfg["regex_constraints"].values())
        validator_chain.set_next(RegexValidator(regex_cols, regex_patterns, logger))
    
    if not validator_chain.validate(df):
        raise ValueError("Validation failed. Check logs.")
    
    # Step 3: Categorize
    if config.get("categorization", {}).get("enabled", True):
        input_csv = "temp_input.csv"
        output_categorized_csv = "temp_categorized.csv"
        output_final_csv = "temp_final.csv"
        df.to_csv(input_csv, index=False)
        categorize_complaints(input_csv, output_categorized_csv)
        post_process_data(output_categorized_csv, output_final_csv)
        df = pd.read_csv(output_final_csv)
    
    # Step 4: Sentiment Analysis
    if config.get("sentiment_analysis", {}).get("enabled", True):
        classifiers = {
            "BERT": BERTClassifier(),
            "VADER": VaderSentimentClassifier(),
            "DistilRoberta Emotion": DistilRobertaClassifier(),
            "Roberta Emotion": RobertaClassifier(),
        }
        
        selected_classifiers = config["sentiment_analysis"].get("classifiers", classifiers.keys())

        for name in selected_classifiers:
            classifier = classifiers.get(name)
            if classifier:
                context = SentimentAnalysisContext(classifier)
                df = context.analyze(df, text_cols=["title_with_desc"])
    
    # Step 5: Save output
    output_path = config.get("output", {}).get("save_path", "output/results.json")
    df.to_json(output_path, orient="records", indent=4)
    
    return df

# Example usage
df = pd.read_csv("files/sentiment_scored_2023_data.csv")  # Load your dataset
result_df = preprocess_validate_categorize_analyze(df)
