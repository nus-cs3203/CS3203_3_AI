import pandas as pd

from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.emotion.distilroberta import DistilRobertaClassifier
from sentiment_analyser.emotion.roberta import RobertaClassifier
from sentiment_analyser.polarity.advanced import AdvancedSentimentClassifier
from sentiment_analyser.polarity.bert import BERTClassifier
from sentiment_analyser.polarity.custom import CustomSentimentClassifier
from sentiment_analyser.polarity.vader import VaderSentimentClassifier

# Load dataset
df = pd.read_csv("files/filtered_complaints.csv").sample(1000, random_state=1)
df["title_with_desc"] = df["title"] + " " + df["description"]


# List of classifiers to test
classifiers = [
    #("BERT", BERTClassifier()),
    ("VADER", VaderSentimentClassifier()),
    #("Advanced", AdvancedSentimentClassifier()),
    #("Customed", CustomSentimentClassifier()),
    #("DistilRoberta Emotion", DistilRobertaClassifier()),
    #("Roberta Emotion", RobertaClassifier()),
]

# Run polarity classifiers one by one with clear output
for name, classifier in classifiers:
    print(f"\n===== Running {name} Sentiment Analysis =====")
    context = SentimentAnalysisContext(classifier)
    df = context.analyze(df, text_cols=["title_with_desc"])
    
    # Print results
    columns_to_print = [
        "title_with_desc_score",
        "title_with_desc_label",
        "title_with_desc_emotion",
        "title_with_desc_confidence"
    ]

    # Filter only the existing columns
    existing_columns = [col for col in columns_to_print if col in df.columns]

    # Print only if there are valid columns
    if existing_columns:
        print(df[existing_columns])

# Save the results
df.to_csv("files/sentiment_results_singlish_robust.csv", index=False)
