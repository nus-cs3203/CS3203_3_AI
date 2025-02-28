import pandas as pd
from common_components.data_preprocessor.concrete_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector

# File paths
INPUT_FILE = "files/sentiment_scored_2023_data.csv"
OUTPUT_FILE = "files/preprocessed_data.csv"

# Define critical and text columns
CRITICAL_COLUMNS = ["title", "selftext"]
TEXT_COLUMNS = ["title", "selftext"]

def main():
    """Loads data, applies preprocessing, and saves the processed dataset."""
    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    # Create builder and director
    builder = GeneralPreprocessorBuilder(CRITICAL_COLUMNS, TEXT_COLUMNS, df)
    director = PreprocessingDirector(builder)
    director.construct_general_builder()  

    # Get processed data
    processed_df = builder.get_result()  # Retrieve processed DataFrame

    # Save the processed file
    processed_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Preprocessed data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
