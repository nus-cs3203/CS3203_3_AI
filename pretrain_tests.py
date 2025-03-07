import pandas as pd

from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.emotion.distilroberta import DistilRobertaClassifier
from sentiment_analyser.emotion.roberta import RobertaClassifier
from sentiment_analyser.polarity.bert import BERTClassifier
from sentiment_analyser.polarity.vader import VaderSentimentClassifier

# Load dataset
df = pd.read_csv("files/sentiment_scored_2023_data.csv").head(100)

# Define critical and text columns
CRITICAL_COLUMNS = ["title"]
TEXT_COLUMNS = ["title_with_desc", "comments"]
SUBSET_COLUMNS = ["title", "selftext"]

# Step 1: Preprocessing
builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, text_columns=TEXT_COLUMNS, data=df, subset=SUBSET_COLUMNS)
director = PreprocessingDirector(builder)
director.construct_builder()
df = builder.get_result()

# Use top 5 rows
print("\n===== Original DataFrame =====")
df_shape_expected = df.shape

# List of classifiers to test
classifiers = [
    ("BERT", BERTClassifier()),
    ("VADER", VaderSentimentClassifier()),
    ("DistilRoberta Emotion", DistilRobertaClassifier()),
    ("Roberta Emotion", RobertaClassifier()),
]

# Run polarity classifiers one by one with clear output
for name, classifier in classifiers:
    print(f"\n===== Running {name} Sentiment Analysis =====")
    context = SentimentAnalysisContext(classifier)
    df = df.copy()
    df = context.analyze(df, text_cols=["title_with_desc"])
    
    # Assert that the DataFrame shape has 2 more columns and the same number of rows
    df_new_shape = df.shape
    print(f"Shape before: {df_shape_expected}, Shape after: {df_new_shape}")
    print(df.columns[-4:])
    assert df_new_shape[0] == df_shape_expected[0], "Number of rows should remain the same"
    if name != "DistilRoberta Emotion" and name != "Roberta Emotion":
        assert df_new_shape[1] == df_shape_expected[1] + 2, "Number of columns should increase by 2"
    else:
        assert df_new_shape[1] == df_shape_expected[1] + 4, "Number of columns should increase by 4"

    # Check new columns
    new_columns = df.columns[-2:]
    for col in new_columns:
        if "score" in col:
            assert df[col].between(-1, 1).all(), f"Values in {col} should be between -1 and 1"
        elif "confidence" in col:
            assert df[col].between(0, 1).all(), f"Values in {col} should be between 0 and 1"
        elif "label" in col or "emotion" in col:
            assert df[col].dtype == object, f"Values in {col} should be strings"
        else:
            raise ValueError(f"Unexpected column name: {col}")