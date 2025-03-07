import pandas as pd
import pytest
from common_components.data_preprocessor.concrete_advanced_builder import AdvancedPreprocessorBuilder
from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.concrete_minimal_builder import MinimalPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector


# Input dataset
input_data = pd.read_csv( "tests/preprocessor_validator/data/input.csv")

# Expected output paths
EXPECTED_OUTPUT_FILES = [
    "tests/preprocessor_validator/data/expected_minimal.csv",
    "tests/preprocessor_validator/data/expected_general.csv",
    "tests/preprocessor_validator/data/expected_advanced.csv",
]

expected_minimal = pd.read_csv(EXPECTED_OUTPUT_FILES[0])
expected_general = pd.read_csv(EXPECTED_OUTPUT_FILES[1])
expected_advanced = pd.read_csv(EXPECTED_OUTPUT_FILES[2])


# Define columns for processing
CRITICAL_COLUMNS = ["title"]
TEXT_COLUMNS = ["title", "selftext", "comments"]
SUBSET_COLUMNS = ["title", "selftext"]

# MINIMAL PREPROCESSOR TEST

builder = MinimalPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, data=input_data, text_columns=TEXT_COLUMNS, subset=SUBSET_COLUMNS)
director = PreprocessingDirector(builder)
director.construct_builder()
output_minimal = builder.get_result()

# Compare processed data with expected output
try:
    output_minimal.equals(expected_minimal)
    print("Test passed for:", "GeneralPreprocessorBuilder")
except AssertionError as e:
    print("Test failed for:", "GeneralPreprocessorBuilder")
    print(e)


# GENERAL PREPROCESSOR TEST

builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, data=input_data, subset=SUBSET_COLUMNS, text_columns=TEXT_COLUMNS)
director = PreprocessingDirector(builder)
director.construct_builder()
output_general = builder.get_result()

# Compare processed data with expected output
try:
    output_general.equals(expected_general)
    print("Test passed for:", "GeneralPreprocessorBuilder")
except AssertionError as e:
    print("Test failed for:", "GeneralPreprocessorBuilder")
    print(e)


# ADVANCED PREPROCESSOR TEST

builder = AdvancedPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, data=input_data, subset=SUBSET_COLUMNS, text_columns=TEXT_COLUMNS)
director = PreprocessingDirector(builder)
director.construct_builder()
output_advanced = builder.get_result()

# Compare processed data with expected output
try:
    output_advanced.equals(expected_advanced)
    print("Test passed for:", "AdvancedPreprocessorBuilder")
except AssertionError as e:
    print("Test failed for:", "AdvancedPreprocessorBuilder")
    print(e)

