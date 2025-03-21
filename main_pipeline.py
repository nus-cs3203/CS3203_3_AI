import pandas as pd

from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.length_validator import LengthValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.validator_logger import ValidatorLogger
from categorizer.r1_categorizer import categorize_complaints
from categorizer.post_process_data import post_process_data
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

# Step 2: Validation
logger = ValidatorLogger()
validator_chain = (
    NotEmptyValidator(CRITICAL_COLUMNS, logger)
    .set_next(OnlyStringValidator(TEXT_COLUMNS, logger))
)

validator_chain.validate(df)

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

print("\n===== Final DataFrame Processed Successfully =====")
print(df.head())
print(df.columns)
df.to_csv("csv_results/final_processed_data.csv", index=False)
