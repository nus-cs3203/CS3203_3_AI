import pandas as pd

from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.emotion.distilroberta import DistilRobertaClassifier
from sentiment_analyser.emotion.roberta import RobertaClassifier
from sentiment_analyser.polarity.bert import BERTClassifier
from sentiment_analyser.polarity.vader import VaderSentimentClassifier

# Load dataset
df = pd.read_csv("files/sentiment_scored_2023_data.csv").head(100)
df.dropna(subset=["title", "selftext"], inplace=True)
df.reset_index(drop=True, inplace=True)
df["title_with_desc"] = df["title"] + " " + df["selftext"]

# Use top 5 rows
df_sample = df.head(5)
print("\n===== Original DataFrame =====")
print(df_sample[["title_with_desc"]])

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
    df_sample = context.analyze(df_sample, text_cols=["title_with_desc"])
    
    # Print results
    columns_to_print = [
        "title_with_desc_score",
        "title_with_desc_label",
        "title_with_desc_emotion",
        "title_with_desc_confidence"
    ]

    # Filter only the existing columns
    existing_columns = [col for col in columns_to_print if col in df_sample.columns]

    # Print only if there are valid columns
    if existing_columns:
        print(df_sample[existing_columns])

print("\n===== Final DataFrame with All Classifications =====")
print(df_sample)
