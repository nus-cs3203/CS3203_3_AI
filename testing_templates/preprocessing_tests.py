import pandas as pd
import pytest
from common_components.data_preprocessor.concrete_builder_advanced import AdvancedPreprocessorBuilder
from common_components.data_preprocessor.concrete_builder_general import GeneralPreprocessorBuilder
from common_components.data_preprocessor.concrete_builder_minimal import MinimalPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector

# Sample test dataset paths
TEST_INPUT_FILES = [
    "test_data/preprocess_1.csv",
    "test_data/preprocess_2.csv",
    "test_data/preprocess_3.csv",
]
EXPECTED_OUTPUT_FILES = [
    "expected_data/preprocess_1.csv",
    "expected_data/preprocess_2.csv",
    "expected_data/preprocess_3.csv",
]

CRITICAL_COLUMNS = ["title", "selftext"]
TEXT_COLUMNS = ["title", "selftext"]

@pytest.mark.parametrize("builder_cls, input_file, expected_file", [
    (GeneralPreprocessorBuilder, "test_data/preprocess_1.csv", "expected_data/preprocess_2.csv"),
    (AdvancedPreprocessorBuilder, "test_data/no_text.csv", "expected_data/no_text_processed.csv"),
    (MinimalPreprocessorBuilder, "test_data/preprocess_3.csv", "expected_data/preprocess_3.csv"),
])
def test_preprocessing_pipeline(builder_cls, input_file, expected_file):
    """Test preprocessing pipelines with different builders against expected outputs."""
    # Load test dataset
    df = pd.read_csv(input_file)
    expected_df = pd.read_csv(expected_file)

    # Initialize builder and director
    builder = builder_cls(CRITICAL_COLUMNS, TEXT_COLUMNS, df)
    director = PreprocessingDirector(builder)
    director.construct_general_builder()

    # Get processed data
    processed_df = builder.get_result()

    # Compare processed data with expected output
    pd.testing.assert_frame_equal(processed_df, expected_df, check_dtype=False)
