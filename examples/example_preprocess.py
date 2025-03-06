import pandas as pd

from common_components.data_preprocessor.concrete_advanced_builder import AdvancedPreprocessorBuilder
from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.concrete_minimal_builder import MinimalPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector

# Load dataset
df = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Define critical and text columns
CRITICAL_COLUMNS = ["title"]
TEXT_COLUMNS = ["title_with_desc", "comments"]
SUBSET_COLUMNS = ["title", "selftext"]

# Step 1: Preprocessing
builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, text_columns=TEXT_COLUMNS, data=df, subset=SUBSET_COLUMNS)
director = PreprocessingDirector(builder)
director.construct_builder()
processed_df = builder.get_result()

# Save the processed file
processed_df.to_csv("files/preprocessed_data.csv", index=False)
