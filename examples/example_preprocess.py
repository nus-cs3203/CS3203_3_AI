import pandas as pd

from common_components.data_preprocessor.concrete_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector

# Load dataset
df = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Define critical and text columns
CRITICAL_COLUMNS = ["title", "selftext"]
TEXT_COLUMNS = ["title", "selftext"]

# Create builder
builder = GeneralPreprocessorBuilder(CRITICAL_COLUMNS, TEXT_COLUMNS)

# Use director to execute preprocessing
director = PreprocessingDirector(builder)
processed_df = director.construct(df)

# Save the processed file
processed_df.to_csv("/Users/aishwaryahariharaniyer/Desktop/CS3203_3_AI/files/preprocessed_data.csv", index=False)
