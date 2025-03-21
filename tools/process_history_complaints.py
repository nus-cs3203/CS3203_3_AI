import pandas as pd
import os

# Import necessary functions
from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.validator_logger import ValidatorLogger
from categorizer.post_process_data import post_process_data
from categorizer.r1_categorizer import categorize_complaints  
from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.emotion.distilroberta import DistilRobertaClassifier
from sentiment_analyser.emotion.roberta import RobertaClassifier
from sentiment_analyser.polarity.bert import BERTClassifier
from sentiment_analyser.polarity.vader import VaderSentimentClassifier


def process_complaints(file_path: str):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        # df = df.head(1000)
        df["title_with_desc"] = df["title"] + " " + df["selftext"]

        # Define critical and text columns
        CRITICAL_COLUMNS = ["title_with_desc"]
        TEXT_COLUMNS = ["title_with_desc", "comments"]

        # Debugging line to print the DataFrame after fetching data
        print("DataFrame after fetching data:", df)

        # Add an empty column named 'comments'
        df['comments'] = ''

        # Rename the 'name' column to 'id'
        df.rename(columns={'name': 'id'}, inplace=True)

        # Debugging line to print the DataFrame after adding 'comments' and renaming 'name'
        print("DataFrame after adding 'comments' and renaming 'name':", df)

        # # Preprocessing
        # builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, text_columns=TEXT_COLUMNS, data=df)
        # director = PreprocessingDirector(builder)
        # director.construct_builder()
        # df = builder.get_result()

        # # Debugging line to print the DataFrame after preprocessing
        # print("DataFrame after preprocessing:", df)

        # Validation
        logger = ValidatorLogger()
        validator_chain = (
            NotEmptyValidator(CRITICAL_COLUMNS, logger)
            .set_next(OnlyStringValidator(TEXT_COLUMNS, logger))
        )
        validation_result = validator_chain.validate(df)
        if not validation_result["success"]:
            raise ValueError(f"Validation failed: {validation_result['errors']}")

        # Debugging line to print the DataFrame after validation
        print("DataFrame after validation:", df)

        # Categorization
        categories = [
            "Housing", "Healthcare", "Public Safety", "Transport",
            "Education", "Environment", "Employment", "Public Health",
            "Legal", "Economy", "Politics", "Technology",
            "Infrastructure", "Others"
        ]
        
        df = categorize_complaints(df=df, categories=categories)

        # Debugging line to print the DataFrame after categorization
        print("DataFrame after categorization:", df)

        # Sentiment Analysis
        classifiers = [
            # ("BERT", BERTClassifier()),
            ("VADER", VaderSentimentClassifier()),
            # ("DistilRoberta Emotion", DistilRobertaClassifier()),
            # ("Roberta Emotion", RobertaClassifier()),
        ]

        for name, classifier in classifiers:
            print(f"\n===== Running {name} Sentiment Analysis =====")
            context = SentimentAnalysisContext(classifier)
            df = context.analyze(df, text_cols=["title_with_desc"])

        # Debugging line to print the DataFrame after sentiment analysis
        print("DataFrame after sentiment analysis:", df)

        output_csv = "last_round_files/sentiment_analysis_result_before_post_processing.csv"
        df.to_csv(output_csv, index=False)
        # Post-processing
        df = post_process_data(df=df)

        # Debugging line to print the DataFrame after post-processing
        print("DataFrame after post-processing:", df)

        # Return the processed DataFrame as CSV
        output_csv = "last_round_files/all_complaints_2022_2025.csv"
        df.to_csv(output_csv, index=False)
        print("Sentiment analysis completed successfully.")
        print(f"Results saved to {output_csv}")

    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("Error occurred:", error_detail)


def filter_complaints(input_file: str, output_file: str):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter the DataFrame
    filtered_df = df[(df['Sentiment Score'] < -0.5) & (df['Confidence Score'] > 0.8)]
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    file_path = "last_round_files/all_posts_2022_2025.csv"  # Replace with your actual file path
    process_complaints(file_path)

    # Example usage for filtering complaints
    input_file = "last_round_files/all_complaints_2022_2025.csv"
    output_file = "last_round_files/filtered_complaints.csv"
    filter_complaints(input_file, output_file)